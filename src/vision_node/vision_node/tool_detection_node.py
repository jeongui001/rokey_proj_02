"""

pyrealsense2로 카메라를 직접 잡고(realsense2_camera 드라이버 불필요) YOLO 검출부터
뎁스 융합·yaw 계산까지 수행해 두 토픽으로 발행한다:

- /detection/tool_boxes (DetectionArray): 프레임별 2D bbox 검출 전체
- /vision/tool_track (ToolTrack): 추적 대상 1개의 위치 xyz + yaw + timestamp + depth_valid
  (속도는 칼만 필터가 계산하므로 발행하지 않는다)

좌표계: watch_posx 파라미터를 주면 base_link 좌표(mm 단위 posx와 hand-eye로 합성),
비워두면 카메라 광학 좌표계(m)로 발행한다. frame_id로 구분된다.

주의: 기존 vision_node(외부 검출 구독 + TF 실시간 조회 방식)와 같은 토픽에 발행하므로
둘을 동시에 띄우면 안 된다. 이 노드는 하드웨어 TF 연동 전의 단독 실행용이다.
"""
import os

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node

from handover_interfaces.msg import Detection2D, DetectionArray, ToolTrack

from vision_node.grasp_geometry import (
    AxisSmoother, is_bbox_at_edge, patch_median_depth, posx_to_matrix,
    tool_axis_from_depth, yaw_deg_to_quaternion,
)

DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 2.0
PATCH_HALF = 4
YAW_MIN_MASK_PX = 50
FOV_MARGIN_PX = 8
# 장단축비(길이/폭)가 이 값 이상이면 PCA 각도를 완전히 신뢰. 정사각형에 가까울수록
# (렌치/망치 머리처럼 폭이 넓은 공구) 각도가 노이즈에 민감해 튀므로 신뢰도를 낮춘다.
ELONGATION_TRUST_MIN = 1.3
ELONGATION_ALPHA_FLOOR = 0.2  # 저신뢰 구간에서 스무딩 alpha에 곱할 최소 배율

# 클래스 구분용 색상 팔레트 (BGR). cls_id 순서대로 순환 배정 - 프로토타입과 동일
CLASS_COLORS = [
    (0, 255, 0),    # 초록
    (255, 100, 0),  # 파랑-청록
    (0, 0, 255),    # 빨강
    (0, 255, 255),  # 노랑
    (255, 0, 255),  # 마젠타
    (255, 255, 0),  # 시안
]


def _default_resource(name):
    """colcon 설치 경로가 있으면 share/resource, 없으면(미빌드 개발 중) 소스 트리 경로."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('vision_node'), 'resource', name)
        if os.path.exists(path):
            return path
    except Exception:
        pass
    return os.path.join(os.path.dirname(__file__), '..', 'resource', name)


class ToolDetectionNode(Node):
    """RealSense 직접 캡처 + YOLO + 뎁스 융합 + 그립 yaw를 한 노드로 묶은 단독 실행형."""

    def __init__(self):
        super().__init__('tool_detection_node')
        self.declare_parameter('model_path', _default_resource('tool_detector_best.pt'))
        self.declare_parameter('hand_eye_path', _default_resource('T_gripper2camera.npy'))
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('tool_class', '')       # ''=최고 score 검출 추적, 지정 시 그 클래스만
        self.declare_parameter('watch_posx', [0.0])    # 관찰 자세 posx 6원소(mm, ZYZ deg). 기본(6원소 아님)=카메라 좌표 발행
        self.declare_parameter('yaw_depth_band_m', 0.008)  # 공구 윗면에서 이보다 깊은 픽셀은 벨트로 보고 제외
        self.declare_parameter('axis_smooth_alpha', 0.25)
        self.declare_parameter('depth_valid_min_ratio', 0.2)  # 패치 유효 비율이 이 미만이면 depth_valid=False
        self.declare_parameter('show_window', False)   # 개발 확인용 OpenCV 창 (프로토타입과 같은 화면)

        self.conf = self.get_parameter('conf').value
        self.tool_class = self.get_parameter('tool_class').value
        self.yaw_band = self.get_parameter('yaw_depth_band_m').value
        self.valid_min_ratio = self.get_parameter('depth_valid_min_ratio').value
        self.show_window = self.get_parameter('show_window').value

        # base <- camera 변환 (단계 6과 동일): 6원소 posx가 주어졌을 때만 활성
        watch_posx = list(self.get_parameter('watch_posx').value)
        self.T_base2cam = None
        if len(watch_posx) == 6:
            T_g2c = np.load(self.get_parameter('hand_eye_path').value)
            self.T_base2cam = posx_to_matrix(watch_posx) @ T_g2c
            self.frame_id = 'base_link'
            self.get_logger().info(f'베이스 변환 활성: watch_posx={watch_posx}')
        else:
            self.frame_id = 'camera_color_optical_frame'
            self.get_logger().info('watch_posx 미설정 - 카메라 광학 좌표계(m)로 발행')

        from ultralytics import YOLO  # import가 느려서(수 초) 노드 초기화 시점에 수행
        self.model = YOLO(self.get_parameter('model_path').value)
        self.class_colors = {name: CLASS_COLORS[i % len(CLASS_COLORS)]
                             for i, name in self.model.names.items()}
        self.smoother = AxisSmoother(alpha=self.get_parameter('axis_smooth_alpha').value)
        self.last_valid_z = None  # depth 무효 프레임에서 z 유지 (전체 계획.md 2.7절과 같은 방침)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.pub_detections = self.create_publisher(DetectionArray, '/detection/tool_boxes', 10)
        self.pub_tool_track = self.create_publisher(ToolTrack, '/vision/tool_track', 10)
        # poll_for_frames는 논블로킹이라 타이머를 카메라 fps보다 촘촘히 돌려도 안전
        self.timer = self.create_timer(1.0 / 60.0, self._on_timer)

    def _on_timer(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        frame = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()) * self.depth_scale
        stamp = self.get_clock().now().to_msg()

        result = self.model.predict(source=frame, conf=self.conf, verbose=False)[0]

        det_msg = DetectionArray()
        det_msg.header.stamp = stamp
        det_msg.header.frame_id = self.frame_id
        detections = []  # (class_name, score, x1, y1, x2, y2)
        for box in result.boxes:
            d = Detection2D()
            d.class_name = result.names[int(box.cls[0])]
            d.score = float(box.conf[0])
            d.x1, d.y1, d.x2, d.y2 = map(int, box.xyxy[0])
            det_msg.detections.append(d)
            detections.append(d)
        self.pub_detections.publish(det_msg)

        # 이번 프레임에 아예 안 보이는 클래스는 이력을 지운다 - 물체가 화면에서 완전히
        # 사라졌다 다시 나타났을 때 이전 이력과 섞이지 않게.
        self.smoother.reset_missing({d.class_name for d in detections})

        # 프로토타입과 동일하게 화면에 보이는 모든 검출에 대해 뎁스/yaw를 매 프레임 계산한다.
        # 발행 대상 1건에만 계산하면, 물체 여러 개의 score가 근소한 차이로 매 프레임 뒤바뀔 때
        # (1) 다른 클래스는 그 프레임 동안 스무딩이 갱신되지 않아 나중에 값이 튀고
        # (2) 시각화 축이 프레임마다 다른 물체로 옮겨 다녀 흔들리는 것처럼 보인다.
        infos = [self._compute_detection_info(d, depth_m) for d in detections]

        target_idx = self._pick_target_idx(detections)
        if target_idx is not None:
            track = self._make_tool_track(detections[target_idx], infos[target_idx], stamp)
            if track is not None:
                self.pub_tool_track.publish(track)

        if self.show_window:
            self._draw_debug(frame, depth_m, detections, infos)

    def _pick_target_idx(self, detections):
        """tool_class가 지정돼 있으면 그 클래스 중 최고 score, 아니면 전체 최고 score."""
        candidates = [(i, d) for i, d in enumerate(detections)
                      if not self.tool_class or d.class_name == self.tool_class]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p[1].score)[0]

    def _compute_detection_info(self, det, depth_m):
        """검출 1건의 패치 median 뎁스+deprojection(단계 3)과 마스크+모멘트 yaw(단계 4)를 계산.
        스무딩 상태(self.smoother)가 클래스별로 매 프레임 갱신되도록 모든 검출에 대해 호출한다."""
        h, w = depth_m.shape
        cx = min(max((det.x1 + det.x2) // 2, 0), w - 1)
        cy = min(max((det.y1 + det.y2) // 2, 0), h - 1)
        z_m, valid_ratio = patch_median_depth(
            depth_m, cx, cy, half=PATCH_HALF, dmin=DEPTH_MIN_M, dmax=DEPTH_MAX_M)

        info = {
            'cx': cx, 'cy': cy, 'z_m': z_m, 'valid_ratio': valid_ratio,
            'X': None, 'Y': None, 'Z': None,
            'edge': False, 'axis_deg': None, 'grip_deg': None, 'rect': None,
            'origin': (max(det.x1, 0), max(det.y1, 0)),
        }
        if z_m is not None:
            info['X'], info['Y'], info['Z'] = rs.rs2_deproject_pixel_to_point(
                self.intrinsics, [cx, cy], z_m)

        # bbox가 화면 가장자리에 가까우면 잘렸을 수 있어 정보성 플래그로만 남긴다 - 컨베이어
        # 위 공구는 시야 가장자리에 걸치는 일이 흔해서, 닿을 때마다 무효 처리하면 축이
        # 거의 안 나온다. 대신 보이는 부분만으로도 계산은 계속 시도한다.
        info['edge'] = is_bbox_at_edge(det.x1, det.y1, det.x2, det.y2, w, h, FOV_MARGIN_PX)
        ox, oy = max(det.x1, 0), max(det.y1, 0)
        roi = depth_m[oy:min(det.y2, h), ox:min(det.x2, w)]
        intr = self.intrinsics
        axis_deg, rect, elongation = tool_axis_from_depth(
            roi, intr.fx, intr.fy, intr.ppx, intr.ppy, ox=ox, oy=oy,
            dmin=DEPTH_MIN_M, dmax=DEPTH_MAX_M,
            band_m=self.yaw_band, min_px=YAW_MIN_MASK_PX)
        if axis_deg is not None:
            trust = min(1.0, max(0.0, (elongation - 1.0) / (ELONGATION_TRUST_MIN - 1.0)))
            alpha = self.smoother.alpha * max(ELONGATION_ALPHA_FLOOR, trust)
            axis_deg = self.smoother.update(det.class_name, axis_deg, alpha=alpha)
            info['axis_deg'] = axis_deg
            info['grip_deg'] = (axis_deg + 90) % 180  # top-down 파지: 장축에 수직으로 닫음
            info['rect'] = rect
        return info

    def _make_tool_track(self, det, info, stamp):
        """검출 1건 + 사전 계산된 info -> ToolTrack. 단계 6(베이스 변환)을 수행한다."""
        z_m = info['z_m']
        depth_valid = z_m is not None and info['valid_ratio'] >= self.valid_min_ratio

        # depth 무효 구간은 마지막 유효 z 유지 - 아예 없으면 이번 프레임은 발행 불가
        if z_m is not None:
            self.last_valid_z = z_m
            X, Y, Z = info['X'], info['Y'], info['Z']
        elif self.last_valid_z is not None:
            z_m = self.last_valid_z
            X, Y, Z = rs.rs2_deproject_pixel_to_point(
                self.intrinsics, [info['cx'], info['cy']], z_m)
        else:
            return None

        yaw_deg = info['grip_deg']  # 카메라 좌표계 그립 yaw (base 회전 반영 전)

        track = ToolTrack()
        track.header.stamp = stamp
        track.header.frame_id = self.frame_id
        track.tool_class = det.class_name
        track.confidence = det.score
        track.depth_valid = bool(depth_valid)
        track.approaching = False  # 접근 판정은 속도 추정(정의 측 칼만) 통합 시 채움 - 흐름도 12

        if self.T_base2cam is not None:
            # 카메라 좌표(m) -> mm로 바꿔 베이스 좌표계로 (hand-eye/posx가 mm 단위)
            bx, by, bz, _ = self.T_base2cam @ np.array([X * 1000.0, Y * 1000.0, Z * 1000.0, 1.0])
            track.pose.position.x, track.pose.position.y, track.pose.position.z = bx, by, bz
            if yaw_deg is not None:
                # 그립축 방향벡터(이미지 평면)를 베이스 좌표계로 회전 (top-down 전제)
                d_cam = np.array([np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg)), 0.0])
                d_base = self.T_base2cam[:3, :3] @ d_cam
                yaw_deg = np.degrees(np.arctan2(d_base[1], d_base[0])) % 180
        else:
            track.pose.position.x, track.pose.position.y, track.pose.position.z = X, Y, Z

        if yaw_deg is not None:
            qx, qy, qz, qw = yaw_deg_to_quaternion(yaw_deg)
            track.pose.orientation.x = qx
            track.pose.orientation.y = qy
            track.pose.orientation.z = qz
            track.pose.orientation.w = qw
        else:
            track.pose.orientation.w = 1.0  # yaw 미확정 - identity (task 쪽은 depth_valid와 별개로 처리)
        return track

    def _draw_debug(self, frame, depth_m, detections, infos):
        """프로토타입(yolo_predict_camera.py)과 같은 화면: 모든 검출에 클래스별 색 bbox +
        뎁스/좌표 텍스트 + (가능하면) 흰색 마스크 사각형과 장축선을 그린다."""
        for d, info in zip(detections, infos):
            color = self.class_colors.get(d.class_name, (0, 255, 0))
            if info['X'] is not None:
                pos_text = (f"({info['X']:+.2f},{info['Y']:+.2f},{info['Z']:.2f})m "
                            f"v{info['valid_ratio'] * 100:.0f}%")
            else:
                pos_text = 'no depth'
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
            cv2.circle(frame, (info['cx'], info['cy']), 3, color, -1)
            cv2.putText(frame, f'{d.class_name} {d.score * 100:.0f}%', (d.x1, d.y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, pos_text, (d.x1, d.y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if info['rect'] is not None:
                ox, oy = info['origin']
                rect = info['rect']
                (rcx, rcy), (rw, rh), _ = rect
                box_pts = (cv2.boxPoints(rect)
                           + np.array([ox, oy], dtype=np.float32)).astype(np.int32)
                cv2.polylines(frame, [box_pts], True, (255, 255, 255), 2)
                theta = np.deg2rad(info['axis_deg'])
                half_len = max(rw, rh) / 2
                gcx, gcy = int(rcx) + ox, int(rcy) + oy
                dx, dy = int(half_len * np.cos(theta)), int(half_len * np.sin(theta))
                cv2.line(frame, (gcx - dx, gcy - dy), (gcx + dx, gcy + dy), (255, 255, 255), 2)
                yaw_text = f"axis {info['axis_deg']:.0f} grip {info['grip_deg']:.0f}"
                if info['edge']:
                    yaw_text += ' (edge)'
            elif info['edge']:
                yaw_text = 'yaw invalid (edge, no mask)'
            else:
                yaw_text = ''
            if yaw_text:
                cv2.putText(frame, yaw_text, (d.x1, min(d.y2 + 18, frame.shape[0] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_m / self.depth_scale, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('tool_detection_node', np.hstack((frame, depth_vis)))
        cv2.waitKey(1)

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ToolDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
