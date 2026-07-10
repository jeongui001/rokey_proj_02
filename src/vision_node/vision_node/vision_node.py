import json
import os
import time
from collections import deque

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener, TransformException
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import String

from handover_interfaces.msg import ToolTrack, HandTrack, DetectionArray, VisionTiming
from handover_interfaces.srv import SetVisionMode

from vision_node.tracking import (
    ToolTracker, pixel_to_camera_xyz, transform_to_matrix, camera_to_base, is_approaching,
    KPT_CONF_MIN, detection_center,
)
from vision_node.hand_tracking import create_hands_detector, detect_hand_landmarks, is_fist
from vision_node.grasp_geometry import (
    AxisSmoother, is_bbox_at_edge, patch_median_depth, tool_axis_from_depth,
    yaw_deg_to_quaternion,
)

# hand-eye 캘리브레이션(T_gripper2camera.npy)이 이미 카메라 광학 좌표계 기준이라,
# RealSense가 내부적으로 발행하는 camera_link->camera_color_optical_frame 회전을 또
# 거치면(=color_msg.header.frame_id로 조회하면) 회전이 두 번 걸려 축이 섞인다(x<->z 결합).
# vision_node.launch.py의 static_transform_publisher가 이 이름으로 link_6->(광학좌표계)를
# 직접 발행하므로, 캘리브레이션을 한 번만 적용하려면 이 프레임을 직접 조회해야 한다.
CAMERA_OPTICAL_CALIB_FRAME = 'camera_optical_calib'

# 뎁스/축 계산 상수 - tool_detection_node(프로토타입 계열)에서 실기 검증된 값 그대로.
DEPTH_MAX_M = 2.0        # 이보다 먼 뎁스는 무효(배경/노이즈)
PATCH_HALF = 4           # patch_median_depth 반경 -> (2*4+1)^2 = 9x9 패치
YAW_MIN_MASK_PX = 50     # 공구 윗면 마스크 최소 픽셀 수 - 미달이면 축 계산 포기
FOV_MARGIN_PX = 8        # bbox가 화면 가장자리에 이만큼 가까우면 잘림 의심 플래그
# 장단축비(길이/폭)가 이 값 이상이면 PCA 각도를 완전히 신뢰. 정사각형에 가까울수록
# (렌치/망치 머리처럼 폭이 넓은 공구) 각도가 노이즈에 민감해 튀므로 신뢰도를 낮춘다.
ELONGATION_TRUST_MIN = 1.3
ELONGATION_ALPHA_FLOOR = 0.2  # 저신뢰 구간에서 스무딩 alpha에 곱할 최소 배율

# 디버그 이미지용 클래스 색상 팔레트(BGR) - YOLO 모델 클래스 목록을 모르는 상태(이 노드는
# 검출을 직접 안 함)라 tool_detection_node처럼 순번 배정이 안 되니 이름 해시로 고정 배정한다.
_DEBUG_CLASS_COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
]


def _class_color(class_name):
    return _DEBUG_CLASS_COLORS[hash(class_name) % len(_DEBUG_CLASS_COLORS)]


class VisionNode(Node):
    """RealSense 전담 노드(전체 계획.md 1.4절). set_mode로 지정된 모드에 따라
    공구 추적(TRACK_TOOL) 또는 손 추적(TRACK_HAND)을 수행해 스트리밍한다.

    실제 검출(YOLO)은 이 노드가 하지 않는다 - object_detection(팀원3)이
    /detection/tool_boxes로 bbox를 주면, 이 노드는 그걸 받아 추적+3D 복원만 한다.
    """

    def __init__(self):
        super().__init__('vision_node')
        self.mode = SetVisionMode.Request.OFF
        self.tool_class = ''

        self.declare_parameter('vision.min_z_m', 0.10)         # MinZ - 이보다 가까우면 depth 무효 취급
        self.declare_parameter('vision.approach_ref_x', 0.0)    # approaching 판정 기준점
        self.declare_parameter('vision.approach_ref_y', 0.0)
        self.declare_parameter('vision.tracker_alpha', 0.6)     # ToolTracker 위치 스무딩
        self.declare_parameter('vision.tracker_beta', 0.3)      # ToolTracker 속도 스무딩
        # 축(yaw) 계산 파라미터 - tool_detection_node와 동일 기본값
        self.declare_parameter('vision.yaw_depth_band_m', 0.008)  # 공구 윗면에서 이보다 깊은 픽셀은 벨트로 보고 제외
        self.declare_parameter('vision.axis_smooth_alpha', 0.25)
        self.declare_parameter('vision.depth_valid_min_ratio', 0.2)  # 패치 유효 비율이 이 미만이면 depth_valid=False
        # 개발/모니터링용 bbox+축 오버레이 이미지 발행 여부 (전체 계획.md 4.6절 계약 -
        # operator_gui가 구독 예정). 매 프레임 인코딩 비용이 있으니 필요 없으면 끌 수 있게 파라미터화.
        self.declare_parameter('vision.publish_debug_image', True)
        # 주먹 확정에 필요한 연속 프레임 수 - 매 프레임 판별 결과가 떨려도(관절 각도가
        # 경계값 근처일 때) 한두 프레임 잘못 잡혔다고 바로 로봇을 멈추지 않게 디바운스한다.
        # HandTrack.fist는 이미 이 확인을 거친 "확정된" 값이라는 계약이라(robot_control의
        # HandServoLoop는 재확인 없이 바로 정지 처리), 여기서 debounce를 책임진다.
        self.declare_parameter('vision.fist_confirm_frames', 5)
        # DEBUG_LOG: 실기 디버깅용 구조화 이벤트. 안정화 후 GUI/로그 정책 확정 시 제거 가능.
        self.declare_parameter('debug.publish_events', True)
        self.declare_parameter('debug.log_vision_decisions', False)
        # 프로파일링: "FPS 저하 vs latency 누적" 판별용 구간 타이밍(/perception/timing).
        # timing_csv_path를 주면 프레임당 1줄 CSV도 남긴다(오프라인 분석: tools/analyze_timing.py) -
        # 시퀀스 안정성(miss rate/지터) 지표의 원천 데이터라 실기 검증 런에서는 켜고 돌 것.
        self.declare_parameter('vision.publish_timing', True)
        self.declare_parameter('vision.timing_csv_path', '')
        self.min_z_m = self.get_parameter('vision.min_z_m').value
        self.yaw_band = self.get_parameter('vision.yaw_depth_band_m').value
        self.valid_min_ratio = self.get_parameter('vision.depth_valid_min_ratio').value
        self.publish_debug_image = self.get_parameter('vision.publish_debug_image').value
        self.approach_ref_xy = (
            self.get_parameter('vision.approach_ref_x').value,
            self.get_parameter('vision.approach_ref_y').value,
        )

        self._bridge = CvBridge()  # ROS Image msg <-> numpy 배열(OpenCV) 변환기
        self.tracker = ToolTracker(
            alpha=self.get_parameter('vision.tracker_alpha').value,
            beta=self.get_parameter('vision.tracker_beta').value)
        self.axis_smoother = AxisSmoother(
            alpha=self.get_parameter('vision.axis_smooth_alpha').value)
        self._hands_detector = None  # 지연 생성 (TRACK_HAND 최초 진입 시 create_hands_detector() 호출)
        self._fist_confirm_count = 0  # 연속으로 주먹으로 판별된 프레임 수(_on_set_mode에서 리셋)

        self.pub_tool_track = self.create_publisher(ToolTrack, '/vision/tool_track', 10)  # 서브스크라이버: task_manager, robot_control(servo_pick 중 직접 구독)
        self.pub_hand_track = self.create_publisher(HandTrack, '/vision/hand_track', 10)  # 서브스크라이버: robot_control(handover_approach 중 직접 구독)
        self.pub_debug_image = self.create_publisher(
            CompressedImage, '/vision/debug_image/compressed', 10)  # 서브스크라이버: operator_gui, 모니터링용(rqt_image_view 등)
        self.pub_debug_events = self.create_publisher(String, '/debug/events', 10)
        self.pub_timing = self.create_publisher(VisionTiming, '/perception/timing', 10)  # 서브스크라이버: 팀 모니터링/tools 분석
        self._t = {}                            # 이번 프레임의 구간 타이밍/추적 기록 (콜백마다 리셋)
        self._timing_window = deque(maxlen=100)  # (callback_ms, e2e_ms, infer_ms) rolling 통계용
        self._timing_csv = None                 # timing_csv_path 설정 시 지연 오픈
        self.srv_set_mode = self.create_service(SetVisionMode, '/vision/set_mode', self._on_set_mode)  # 클라이언트: task_manager

        # eye-in-hand라 3D 복원엔 "지금"이 아니라 "이미지가 찍힌 시각"의 flange pose가 필요하다
        # (전체 계획.md 2.4절) - 그래서 TF를 캐시해두고 이미지 stamp로 lookup_transform 한다.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # color/depth/camera_info/검출결과 4개를 같은 시각 근처끼리 묶어서 하나의 콜백으로 받는다.
        # 이래야 "이 bbox가 어느 depth 프레임의 것인지"가 어긋나지 않는다.
        # realsense2_camera 등 카메라 드라이버는 스트림 QoS로 BEST_EFFORT(qos_profile_sensor_data)를
        # 쓰도록 권장되고 버전/설정에 따라 언제든 그렇게 바뀔 수 있다. 여기서 기본 QoS(RELIABLE)를
        # 쓰면 드라이버가 BEST_EFFORT로 바뀌는 순간 에러 없이 프레임을 0개 수신하게 되므로,
        # 카메라 원시 스트림 3개는 명시적으로 맞춰둔다.
        self.sub_color = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw',
            qos_profile=qos_profile_sensor_data)  # 퍼블리셔: realsense2_camera(기성 패키지)
        self.sub_depth = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw',
            qos_profile=qos_profile_sensor_data)  # 퍼블리셔: realsense2_camera(기성 패키지)
        self.sub_info = message_filters.Subscriber(
            self, CameraInfo, '/camera/color/camera_info',
            qos_profile=qos_profile_sensor_data)  # 퍼블리셔: realsense2_camera(기성 패키지)
        self.sub_detections = message_filters.Subscriber(
            self, DetectionArray, '/detection/tool_boxes')  # 퍼블리셔: object_detection(팀원3)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info, self.sub_detections],
            queue_size=10, slop=0.05)
        self._sync.registerCallback(self._on_synced_images)

    def _checkpoint_event(
            self, phase, checkpoint_id, status, message, data=None,
            *, throttle_s=None, log=False):
        """파이프라인 점검.md의 Phase 체크리스트에 대응하는 이벤트를 발행한다."""
        now = time.monotonic()
        key = (checkpoint_id, status)
        if throttle_s is not None:
            last = getattr(self, '_checkpoint_event_last', {}).get(key, 0.0)
            if now - last < throttle_s:
                return
            if not hasattr(self, '_checkpoint_event_last'):
                self._checkpoint_event_last = {}
            self._checkpoint_event_last[key] = now
        payload = {
            'phase': phase,
            'checkpoint_id': checkpoint_id,
            'status': status,
            'message': message,
            'data': data or {},
            'node': self.get_name(),
            'stamp_monotonic': now,
        }
        if bool(self.get_parameter('debug.publish_events').value):
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub_debug_events.publish(msg)
        if log or bool(self.get_parameter('debug.log_vision_decisions').value):
            text = f'[CHECKPOINT][{phase}/{checkpoint_id}] status={status} message={message}'
            if status == 'FAIL':
                self.get_logger().error(text)
            else:
                self.get_logger().info(text)

    # timing CSV 열 순서 (tools/analyze_timing.py와 계약)
    _TIMING_CSV_COLS = (
        'stamp_s', 'infer_ms', 'detect_latency_ms', 'sync_wait_ms', 'tf_ms', 'depth_ms',
        'track_ms', 'yaw_ms', 'debug_ms', 'callback_ms', 'e2e_ms', 'published',
        'n_detections', 'cx', 'cy', 'base_x', 'base_y', 'base_z', 'cam_z',
        'depth_valid', 'grip_deg')

    def _publish_timing(self, color_msg, detection_msg, t_entry, entry_wall_s, published):
        """프레임 하나의 구간 타이밍을 /perception/timing으로 발행하고(팀 실시간 확인용),
        timing_csv_path가 설정돼 있으면 CSV 한 줄을 append한다(오프라인 분석용).

        핵심 진단: callback_ms(콜백 내 연산 합)와 e2e_ms(capture->지금)의 괴리.
        연산 합이 작은데 e2e가 크면 큐 적체(오래된 프레임을 처리 중) - "추론이 느린 게"
        아니라 "밀린 게" 문제라는 뜻이다."""
        if not bool(self.get_parameter('vision.publish_timing').value):
            return
        callback_ms = (time.perf_counter() - t_entry) * 1000.0
        stamp_s = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
        e2e_ms = (self.get_clock().now().nanoseconds * 1e-9 - stamp_s) * 1000.0
        # capture->콜백진입 경과에서 검출 노드 구간을 빼면 "토픽 홉 + 4토픽 정렬 대기".
        # 음수가 나오면 노드 간 시계 불일치 신호라 일부러 clamp하지 않는다.
        sync_wait_ms = (entry_wall_s - stamp_s) * 1000.0 - detection_msg.detect_latency_ms

        msg = VisionTiming()
        msg.header = color_msg.header
        msg.infer_ms = float(detection_msg.infer_ms)
        msg.detect_latency_ms = float(detection_msg.detect_latency_ms)
        msg.sync_wait_ms = float(sync_wait_ms)
        msg.tf_ms = float(self._t.get('tf_ms', 0.0))
        msg.depth_ms = float(self._t.get('depth_ms', 0.0))
        msg.track_ms = float(self._t.get('track_ms', 0.0))
        msg.yaw_ms = float(self._t.get('yaw_ms', 0.0))
        msg.debug_ms = float(self._t.get('debug_ms', 0.0))
        msg.callback_ms = float(callback_ms)
        msg.e2e_ms = float(e2e_ms)
        msg.published = 1 if published else 0
        msg.n_detections = min(len(detection_msg.detections), 255)
        self.pub_timing.publish(msg)

        # rolling(최근 100프레임) 요약을 2초마다 이벤트로 - GUI/콘솔에서 추세 확인용
        self._timing_window.append((callback_ms, e2e_ms, float(detection_msg.infer_ms)))
        if len(self._timing_window) >= 10:
            cbs = sorted(t[0] for t in self._timing_window)
            e2es = sorted(t[1] for t in self._timing_window)
            infs = sorted(t[2] for t in self._timing_window)
            p95 = lambda xs: xs[int(len(xs) * 0.95) - 1]  # noqa: E731
            self.get_logger().info(
                f'[TIMING] callback_ms mean={sum(cbs) / len(cbs):.2f} p95={p95(cbs):.2f} '
                f'e2e_ms mean={sum(e2es) / len(e2es):.2f} p95={p95(e2es):.2f} '
                f'infer_ms mean={sum(infs) / len(infs):.2f} n_frames={len(self._timing_window)}',
                throttle_duration_sec=2.0)

        csv_path = str(self.get_parameter('vision.timing_csv_path').value or '')
        if csv_path:
            row = {
                'stamp_s': f'{stamp_s:.6f}',
                'infer_ms': f'{detection_msg.infer_ms:.3f}',
                'detect_latency_ms': f'{detection_msg.detect_latency_ms:.3f}',
                'sync_wait_ms': f'{sync_wait_ms:.3f}',
                'tf_ms': f"{self._t.get('tf_ms', 0.0):.3f}",
                'depth_ms': f"{self._t.get('depth_ms', 0.0):.3f}",
                'track_ms': f"{self._t.get('track_ms', 0.0):.3f}",
                'yaw_ms': f"{self._t.get('yaw_ms', 0.0):.3f}",
                'debug_ms': f"{self._t.get('debug_ms', 0.0):.3f}",
                'callback_ms': f'{callback_ms:.3f}',
                'e2e_ms': f'{e2e_ms:.3f}',
                'published': int(published),
                'n_detections': len(detection_msg.detections),
                'cx': f"{self._t.get('cx', float('nan')):.1f}",
                'cy': f"{self._t.get('cy', float('nan')):.1f}",
                'base_x': f"{self._t.get('base_x', float('nan')):.4f}",
                'base_y': f"{self._t.get('base_y', float('nan')):.4f}",
                'base_z': f"{self._t.get('base_z', float('nan')):.4f}",
                'cam_z': f"{(self._t.get('cam_z') if self._t.get('cam_z') is not None else float('nan')):.4f}",
                'depth_valid': int(self._t.get('depth_valid', False)),
                'grip_deg': f"{self._t.get('grip_deg', float('nan')):.2f}",
            }
            try:
                if self._timing_csv is None:
                    new_file = not os.path.exists(csv_path)
                    self._timing_csv = open(csv_path, 'a', buffering=1)  # line-buffered
                    if new_file:
                        self._timing_csv.write(','.join(self._TIMING_CSV_COLS) + '\n')
                self._timing_csv.write(','.join(str(row[c]) for c in self._TIMING_CSV_COLS) + '\n')
            except OSError as exc:
                self.get_logger().warn(
                    f'timing CSV 기록에 실패했습니다. (path={csv_path}, error={exc})',
                    throttle_duration_sec=5.0)

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    _SET_MODE_CHECKPOINTS = {
        SetVisionMode.Request.TRACK_TOOL: ('C', 'vision_set_mode_track_tool'),
        SetVisionMode.Request.TRACK_HAND: ('G', 'vision_set_mode_track_hand'),
        SetVisionMode.Request.OFF: ('K', 'vision_set_mode_off'),
    }

    def _on_set_mode(self, request, response):
        """task_manager가 부르는 서비스 핸들러. 모드를 TRACK_TOOL로 새로 켤 때마다
        추적기를 리셋해서 이전 물체(다른 공구, 이전 사이클의 잔상)를 안 물고 가게 한다."""
        self.mode = request.mode
        self.tool_class = request.tool_class
        if request.mode == SetVisionMode.Request.TRACK_TOOL:
            self.tracker.reset()
            self.axis_smoother.reset(request.tool_class)  # 이전 사이클의 축 이력도 함께 초기화
        elif request.mode == SetVisionMode.Request.TRACK_HAND:
            self._fist_confirm_count = 0  # 이전 handover 사이클의 주먹 판정 이력을 지운다
        response.success = True
        response.message = f'mode set to {request.mode} (tool_class={request.tool_class})'
        checkpoint = self._SET_MODE_CHECKPOINTS.get(request.mode)
        if checkpoint is not None:
            phase, checkpoint_id = checkpoint
            self._checkpoint_event(
                phase, checkpoint_id, 'PASS' if response.success else 'FAIL',
                response.message, {'mode': request.mode, 'tool_class': request.tool_class})
        return response

    def _on_synced_images(self, color_msg, depth_msg, info_msg, detection_msg):
        """4개 토픽이 시간적으로 맞춰졌을 때마다(30~60Hz 목표) 호출되는 메인 루프.
        모드에 따라 _track_tool 또는 _track_hand로 위임하고, 결과가 있으면 퍼블리시."""
        t_entry = time.perf_counter()
        entry_wall_s = self.get_clock().now().nanoseconds * 1e-9
        self._t = {}  # 이번 프레임 타이밍 기록 시작 (_track_tool의 구간들이 여기 쌓인다)
        try:
            # "지금"이 아니라 color_msg가 찍힌 시각의 flange pose로 조회 (2.4절 핵심)
            # color_msg.header.frame_id(RealSense가 붙이는 camera_color_optical_frame)가
            # 아니라 CAMERA_OPTICAL_CALIB_FRAME을 조회한다 - 캘리브레이션 회전 중복 적용 방지.
            t0 = time.perf_counter()
            tf_at_stamp = self.tf_buffer.lookup_transform(
                'base_link', CAMERA_OPTICAL_CALIB_FRAME, color_msg.header.stamp,
                timeout=Duration(seconds=0.1))
            self._t['tf_ms'] = (time.perf_counter() - t0) * 1000.0
        except TransformException as ex:
            self.get_logger().warn(
                f'이미지 시각의 camera->base TF 조회에 실패했습니다: {ex}',
                throttle_duration_sec=1.0)
            return

        if self.mode == SetVisionMode.Request.TRACK_TOOL:
            track = self._safe_call(
                self._track_tool, color_msg, depth_msg, info_msg, detection_msg,
                tf_at_stamp, self.tool_class, default=None)
            if track is not None:
                self.pub_tool_track.publish(track)
            self._publish_timing(color_msg, detection_msg, t_entry, entry_wall_s,
                                 published=track is not None)
        elif self.mode == SetVisionMode.Request.TRACK_HAND:
            hand_track = self._safe_call(
                self._track_hand, color_msg, depth_msg, info_msg, tf_at_stamp, default=None)
            if hand_track is not None:
                self.pub_hand_track.publish(hand_track)
        # mode == OFF면 아무것도 안 하고 그냥 리턴 (프레임 버림)

    def _tf_matrix(self, tf_at_stamp):
        """TransformStamped -> tracking.transform_to_matrix가 쓰는 (translation, rotation) 형태로."""
        t = tf_at_stamp.transform.translation
        r = tf_at_stamp.transform.rotation
        return transform_to_matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    def _track_tool(self, color_msg, depth_msg, info_msg, detection_msg, tf_at_stamp, tool_class):
        """저해상도 검출(팀원3 제공) + 3D 복원(tf_at_stamp 사용) + 알파-베타 필터로 ToolTrack을 만든다.

        yaw(그립 방향)는 선택된 bbox의 depth ROI에서 3D 포인트클라우드 PCA로 구한다
        (grasp_geometry.tool_axis_from_depth - 프로토타입 tool_detection_node에서 실기 검증됨).
        """
        # CameraInfo.k는 3x3 intrinsic 행렬을 1차원으로 편 것: [fx,0,ppx, 0,fy,ppy, 0,0,1]
        # numpy 배열이라 float()로 캐스팅해두지 않으면 이후 계산 결과가 numpy 타입으로 오염되어
        # bool 필드(approaching 등)에 대입할 때 타입 에러가 난다.
        fx, fy, ppx, ppy = (float(info_msg.k[0]), float(info_msg.k[4]),
                            float(info_msg.k[2]), float(info_msg.k[5]))
        t0 = time.perf_counter()
        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_m_img = depth_image.astype(np.float64) / 1000.0  # RealSense depth는 보통 mm(16UC1)
        self._t['depth_ms'] = (time.perf_counter() - t0) * 1000.0
        tf_matrix = self._tf_matrix(tf_at_stamp)

        def reconstruct(cx, cy, bbox_w, bbox_h):
            """bbox 중심 픽셀(cx, cy) -> base_link 3D 좌표. ToolTracker.update()가
            후보 bbox마다 이 함수를 호출한다 (tracking.py는 ROS/depth 이미지를 몰라도 되게
            이 클로저 하나로 depth 조회 + intrinsics + tf 변환을 전부 감춘다)."""
            px, py = int(cx), int(cy)
            if not (0 <= py < depth_m_img.shape[0] and 0 <= px < depth_m_img.shape[1]):
                self.get_logger().warn(
                    f'bbox 중심 픽셀({px},{py})이 depth 이미지 범위를 벗어났습니다.',
                    throttle_duration_sec=1.0)
                return None
            # 단일 픽셀은 금속/반사면 뎁스 구멍에 취약해서 patch median으로 보완하되,
            # bbox가 patch(9x9)보다 작으면(멀리 있거나 작은 물체) 패치가 bbox 밖 배경까지
            # 덮어 median이 배경 depth로 쏠릴 수 있어 bbox 안쪽으로 반경을 제한한다.
            half = max(1, min(PATCH_HALF, int(min(bbox_w, bbox_h) // 2)))
            z_m, valid_ratio = patch_median_depth(
                depth_m_img, px, py, half=half, dmin=self.min_z_m, dmax=DEPTH_MAX_M)
            depth_valid = z_m is not None and valid_ratio >= self.valid_min_ratio
            # depth 무효 구간은 마지막 유효 z로 픽셀->광선을 역산해 x,y만 RGB 추적으로 갱신 (2.7절).
            # 추적 사이클의 첫 프레임부터 depth가 무효면(반사면 공구에서 흔함) last_valid_z도
            # 아직 없다 - 이때 z를 0.0 등으로 지어내면 카메라 장착 위치 근방의 엉뚱한 3D 좌표가
            # 나가버린다(2026-07-08 실기 사고: 망치가 보이는데도 로봇이 엉뚱한 위치로 이동해
            # 바닥을 내려찍음 - 원인이 이 좌표 조작이었다). 만들어낼 z가 없으면 이 후보를 그냥
            # 버린다(None) - ToolTracker.update()가 다른 유효 후보를 쓰거나, 없으면 검출 없음과
            # 동일하게 처리한다.
            if z_m is not None:
                z = z_m
            elif self.tracker.last_valid_z is not None:
                z = self.tracker.last_valid_z
            else:
                self.get_logger().warn(
                    f'첫 유효 depth가 없어 3D 좌표 생성을 건너뜁니다 '
                    f'(px={px}, py={py}, valid_ratio={valid_ratio:.3f}).',
                    throttle_duration_sec=1.0)
                return None
            cam_xyz = pixel_to_camera_xyz(px, py, z, fx, fy, ppx, ppy)
            base_xyz = camera_to_base(cam_xyz, tf_matrix)
            return (base_xyz[0], base_xyz[1], base_xyz[2], depth_valid)

        stamp = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
        t0 = time.perf_counter()
        result = self.tracker.update(detection_msg.detections, tool_class, reconstruct, stamp)
        self._t['track_ms'] = (time.perf_counter() - t0) * 1000.0
        if result is None:
            # 검출이 끊긴 프레임 - 축 이력도 지운다. 물체가 화면에서 사라졌다 다시
            # 나타났을 때 이전 물체의 각도가 새 물체 각도와 섞이는 것을 막는다
            # (프로토타입 reset_missing과 같은 방침).
            self.axis_smoother.reset(tool_class)
            if self.publish_debug_image:
                # tool_class 매칭 검출이 없어도 들어온 검출 전체는 그려서 보여준다 -
                # 왜 안 잡히는지(다른 클래스만 보임/아예 없음) 눈으로 바로 확인 가능하게.
                t0 = time.perf_counter()
                self._publish_debug_image(color_msg, detection_msg.detections, None, None)
                self._t['debug_ms'] = (time.perf_counter() - t0) * 1000.0
            self.get_logger().warn(
                f"'{tool_class}' 유효 3D 추적 결과가 없습니다. "
                f"(frame_classes={[d.class_name for d in detection_msg.detections]})",
                throttle_duration_sec=1.0)
            return None  # 이번 프레임엔 tool_class 검출이 없었음 - 퍼블리시 안 함

        position, velocity, depth_valid, chosen_det = result
        self._checkpoint_event(
            'C', 'tool_track_valid', 'PASS',
            'ToolTrack 위치/뎁스/접근 판정이 유효합니다.',
            {
                'tool_class': tool_class,
                'confidence': float(chosen_det.score),
                'depth_valid': bool(depth_valid),
            },
            throttle_s=1.0)
        t0 = time.perf_counter()
        yaw_quat, axis_debug = self._grip_yaw_quaternion(
            chosen_det, depth_m_img, fx, fy, ppx, ppy, tf_matrix, tool_class)
        self._t['yaw_ms'] = (time.perf_counter() - t0) * 1000.0
        # 시퀀스 안정성(지터/miss) 오프라인 분석용 - timing CSV에 같이 실린다
        cx, cy = detection_center(chosen_det)
        self._t.update(cx=cx, cy=cy, base_x=position[0], base_y=position[1],
                       base_z=position[2], depth_valid=bool(depth_valid),
                       # 거리 버킷 분석용 카메라축 거리(마지막 유효 depth) - base_z(로봇 기준 높이)와 다르다
                       cam_z=self.tracker.last_valid_z)
        if axis_debug is not None:
            self._t['grip_deg'] = axis_debug['grip_deg']

        if self.publish_debug_image:
            t0 = time.perf_counter()
            self._publish_debug_image(
                color_msg, detection_msg.detections, chosen_det, axis_debug,
                position=position, depth_valid=depth_valid)
            self._t['debug_ms'] = (time.perf_counter() - t0) * 1000.0

        track = ToolTrack()
        track.header = color_msg.header  # 관측 시각(stamp)은 그대로 - 서보 루프의 시간 정합 기준
        # position/orientation은 위에서 이미 base_link로 변환했으므로 frame_id도 base_link로
        # 고쳐야 한다 - color_msg.header를 그대로 복사하면 frame_id가 카메라 프레임으로 남아
        # robot_control의 _validate_tool_track_message(frame_id=='base_link' 검사)가 거부한다.
        # _track_tool은 TF 조회 성공(_on_synced_images) 후에만 호출되므로 항상 base_link.
        track.header.frame_id = 'base_link'
        track.tool_class = tool_class
        track.confidence = float(chosen_det.score)
        track.kpt0_x = float(chosen_det.kpt0_x)
        track.kpt0_y = float(chosen_det.kpt0_y)
        track.kpt0_conf = float(chosen_det.kpt0_conf)
        track.kpt1_x = float(chosen_det.kpt1_x)
        track.kpt1_y = float(chosen_det.kpt1_y)
        track.kpt1_conf = float(chosen_det.kpt1_conf)
        track.pose.position.x = position[0]
        track.pose.position.y = position[1]
        track.pose.position.z = position[2]
        if yaw_quat is not None:
            (track.pose.orientation.x, track.pose.orientation.y,
             track.pose.orientation.z, track.pose.orientation.w) = yaw_quat
        else:
            self.get_logger().warn(
                f"'{tool_class}' 공구 yaw 축 계산이 불가능해 identity orientation을 사용합니다.",
                throttle_duration_sec=1.0)
            track.pose.orientation.w = 1.0  # 축 미확정 프레임 - identity (구독측은 depth_valid와 별개로 처리)
        track.depth_valid = bool(depth_valid)
        track.approaching = bool(is_approaching(
            (position[0], position[1]), velocity, self.approach_ref_xy))
        return track

    def _grip_yaw_quaternion(self, det, depth_m_img, fx, fy, ppx, ppy, tf_matrix, tool_class):
        """선택된 검출의 bbox depth ROI에서 그립 yaw 쿼터니언(base 기준)을 계산한다.

        장축은 3D 포인트클라우드 PCA(tool_axis_from_depth)로 구하고, 장단축비가 낮을수록
        (정사각형에 가까운 마스크 - PCA 각도가 노이즈에 민감) 스무딩을 강하게 눌러
        저신뢰 관측이 각도를 흔들지 못하게 한다. 그립 방향은 장축에 수직(top-down 파지).
        마스크 픽셀 부족 등으로 축을 못 구한 프레임은 (None, None).

        반환: (쿼터니언 또는 None, 디버그 시각화용 정보 dict 또는 None). 디버그 정보는
        카메라 이미지 평면 좌표계 그대로(base 회전 반영 전)라 _publish_debug_image에서
        원본 프레임 위에 곧바로 그릴 수 있다."""
        # pose 모델 경로: 검출에 keypoint(공구 장축 양 끝점)가 있으면 그 벡터각이 곧
        # 장축이다 - depth 마스크/PCA 없이 직접, 근접·가림에도 bbox보다 강건.
        # keypoint가 없거나 저신뢰면 기존 depth-PCA로 폴백 (box 모델 하위호환).
        kpt_axis = self._axis_from_keypoints(det)
        if kpt_axis is not None:
            axis_deg, kpt_debug, trust = kpt_axis
            alpha = self.axis_smoother.alpha * max(ELONGATION_ALPHA_FLOOR, trust)
            axis_deg = self.axis_smoother.update(tool_class, axis_deg, alpha=alpha)
            grip_deg = (axis_deg + 90.0) % 180.0  # top-down 파지: 장축에 수직으로 닫음
            debug_info = {'kpts': kpt_debug, 'axis_deg': axis_deg, 'grip_deg': grip_deg}
            return self._grip_deg_to_base_quaternion(grip_deg, tf_matrix), debug_info
        # 한쪽 kpt만 유효(근접 시 반대쪽 끝이 화면 밖으로 잘림 - 라이브런 실측)이고
        # 직전까지 keypoint로 축을 추정한 이력이 있으면, 잘린 bbox ROI의 depth-PCA
        # (노이즈로 yaw 드리프트 유발)보다 직전 스무딩 각도를 유지하는 게 정확하다
        # (도구는 접근 중 정지 상태라 축이 변하지 않음).
        c0 = getattr(det, 'kpt0_conf', 0.0)
        c1 = getattr(det, 'kpt1_conf', 0.0)
        if (c0 >= KPT_CONF_MIN) != (c1 >= KPT_CONF_MIN):
            held_deg = self.axis_smoother.current(tool_class)
            if held_deg is not None:
                grip_deg = (held_deg + 90.0) % 180.0
                debug_info = {'axis_deg': held_deg, 'grip_deg': grip_deg, 'held': True}
                return self._grip_deg_to_base_quaternion(grip_deg, tf_matrix), debug_info
        h, w = depth_m_img.shape
        x1, y1 = max(det.x1, 0), max(det.y1, 0)
        x2, y2 = min(det.x2, w), min(det.y2, h)
        if x2 <= x1 or y2 <= y1:
            return None, None
        if is_bbox_at_edge(det.x1, det.y1, det.x2, det.y2, w, h, FOV_MARGIN_PX):
            # 가장자리 걸침은 컨베이어 위에서 흔해 무효화하면 축이 거의 안 나온다 -
            # 보이는 부분만으로 계속 계산하되 정보성 로그만 남긴다 (프로토타입 방침).
            self.get_logger().debug(
                f'{det.class_name} bbox가 화면 가장자리에 걸침 - 잘린 실루엣으로 축 계산',
                throttle_duration_sec=1.0)
        roi = depth_m_img[y1:y2, x1:x2]
        axis_deg, rect, elongation = tool_axis_from_depth(
            roi, fx, fy, ppx, ppy, ox=x1, oy=y1,
            dmin=self.min_z_m, dmax=DEPTH_MAX_M,
            band_m=self.yaw_band, min_px=YAW_MIN_MASK_PX)
        if axis_deg is None:
            self.get_logger().warn(
                f"{det.class_name} depth ROI에서 공구 축을 계산하지 못했습니다. "
                f"(bbox=({det.x1},{det.y1},{det.x2},{det.y2}), min_mask_px={YAW_MIN_MASK_PX})",
                throttle_duration_sec=1.0)
            return None, None
        trust = min(1.0, max(0.0, (elongation - 1.0) / (ELONGATION_TRUST_MIN - 1.0)))
        alpha = self.axis_smoother.alpha * max(ELONGATION_ALPHA_FLOOR, trust)
        axis_deg = self.axis_smoother.update(tool_class, axis_deg, alpha=alpha)
        grip_deg = (axis_deg + 90.0) % 180.0  # top-down 파지: 장축에 수직으로 닫음
        debug_info = {'rect': rect, 'axis_deg': axis_deg, 'grip_deg': grip_deg, 'origin': (x1, y1)}
        return self._grip_deg_to_base_quaternion(grip_deg, tf_matrix), debug_info

    def _axis_from_keypoints(self, det):
        """검출의 2-keypoint(장축 양 끝점)에서 (axis_deg, 디버그 좌표, trust)를 구한다.

        keypoint가 없거나(kpt_conf 기본값 0 - 구 box 모델), 저신뢰거나, 두 점이 너무
        붙어 있으면(각도 신뢰 불가) None - 호출측이 depth-PCA로 폴백한다.
        trust는 두 kpt conf의 평균으로, elongation 기반 trust와 같은 방식으로
        AxisSmoother의 alpha를 누른다(저신뢰 관측이 각도를 흔들지 못하게)."""
        c0, c1 = det.kpt0_conf, det.kpt1_conf
        if c0 < KPT_CONF_MIN or c1 < KPT_CONF_MIN:
            return None
        dx, dy = det.kpt1_x - det.kpt0_x, det.kpt1_y - det.kpt0_y
        # 축 길이 하한: bbox 장변의 30%. 그 미만이면 두 점이 뭉쳐 있어 각도가 노이즈
        # (라벨 스펙상 끝점 간 거리는 장변 근처여야 정상 - validate_pose_labels.py의 50% 경고와 일관)
        min_axis_px = 0.3 * max(det.x2 - det.x1, det.y2 - det.y1)
        if np.hypot(dx, dy) < max(min_axis_px, 2.0):
            self.get_logger().warn(
                f'{det.class_name} keypoint 축이 너무 짧아 depth-PCA로 폴백합니다. '
                f'(axis_px={np.hypot(dx, dy):.1f})',
                throttle_duration_sec=1.0)
            return None
        axis_deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        kpt_debug = ((det.kpt0_x, det.kpt0_y), (det.kpt1_x, det.kpt1_y))
        trust = min(1.0, (c0 + c1) / 2.0)
        return axis_deg, kpt_debug, trust

    def _grip_deg_to_base_quaternion(self, grip_deg, tf_matrix):
        """그립축 방향벡터(카메라 이미지 평면)를 base 좌표계로 회전해 yaw 쿼터니언으로 (top-down 전제)."""
        d_cam = np.array([np.cos(np.deg2rad(grip_deg)), np.sin(np.deg2rad(grip_deg)), 0.0])
        rot = np.array(tf_matrix)[:3, :3]
        d_base = rot @ d_cam
        base_yaw_deg = float(np.degrees(np.arctan2(d_base[1], d_base[0])) % 180.0)
        return yaw_deg_to_quaternion(base_yaw_deg)

    def _publish_debug_image(
            self, color_msg, detections, chosen_det, axis_debug, position=None, depth_valid=None):
        """검출 전체의 bbox + (있으면) 추적 대상의 축선을 그려 압축 이미지로 발행한다
        (전체 계획.md 4.6절 계약, operator_gui/rqt_image_view로 모니터링용). 클래스별
        색상은 YOLO 모델 정보 없이도 이름 해시로 고정 배정한다.

        position/depth_valid가 있으면(= ToolTrack이 실제로 발행된 프레임) 계산된
        base_link 좌표를 화면에 같이 찍는다 - 2026-07-08 실기 사고(z=0 폴백 버그로 엉뚱한
        좌표가 서보 목표가 됨) 이후 추가됨. bbox가 맞게 잡히는지뿐 아니라 좌표 계산
        결과가 말이 되는 값인지(카메라 근처 등 이상값이 아닌지)를 rqt_image_view만 보고도
        바로 판단할 수 있게 하기 위함이다."""
        # 폰트 스케일 주의: 실제 스트림이 424x240(launch 설정)이라 0.5는 글자가 화면을
        # 덮고 서로 겹친다(2026-07-08 실기 확인) - 0.35로 축소하고, base 좌표는 bbox
        # 라벨들과 안 겹치게 화면 하단에 배치한다.
        frame = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8').copy()
        for d in detections:
            color = _class_color(d.class_name)
            is_target = chosen_det is not None and d is chosen_det
            thickness = 3 if is_target else 1
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
            cv2.putText(frame, f'{d.class_name} {d.score * 100:.0f}%', (d.x1, max(d.y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if axis_debug is not None and 'kpts' in axis_debug:
            # keypoint 경로: 끝점 2개(p0 빨강/p1 파랑)와 그 축선을 그대로 그린다
            (p0x, p0y), (p1x, p1y) = axis_debug['kpts']
            p0 = (int(p0x), int(p0y))
            p1 = (int(p1x), int(p1y))
            cv2.line(frame, p0, p1, (255, 255, 255), 1)
            cv2.circle(frame, p0, 3, (0, 0, 255), -1)
            cv2.circle(frame, p1, 3, (255, 0, 0), -1)
            cv2.putText(frame, f"axis {axis_debug['axis_deg']:.0f} grip {axis_debug['grip_deg']:.0f}",
                        (min(p0[0], p1[0]), min(frame.shape[0] - 16, max(p0[1], p1[1]) + 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        elif axis_debug is not None and axis_debug.get('held'):
            # 단일 kpt hold 경로: 그릴 기하가 없어 유지 중인 각도만 텍스트로 표시
            cv2.putText(frame, f"axis {axis_debug['axis_deg']:.0f} grip {axis_debug['grip_deg']:.0f} (hold)",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        elif axis_debug is not None:
            ox, oy = axis_debug['origin']
            rect = axis_debug['rect']
            (rcx, rcy), (rw, rh), _ = rect
            box_pts = (cv2.boxPoints(rect) + np.array([ox, oy], dtype=np.float32)).astype(np.int32)
            cv2.polylines(frame, [box_pts], True, (255, 255, 255), 1)
            theta = np.deg2rad(axis_debug['axis_deg'])
            half_len = max(rw, rh) / 2
            gcx, gcy = int(rcx) + ox, int(rcy) + oy
            dx, dy = int(half_len * np.cos(theta)), int(half_len * np.sin(theta))
            cv2.line(frame, (gcx - dx, gcy - dy), (gcx + dx, gcy + dy), (255, 255, 255), 1)
            # bbox 위 라벨(클래스명)과 겹치지 않게 회전사각형 하단에 표시
            cv2.putText(frame, f"axis {axis_debug['axis_deg']:.0f} grip {axis_debug['grip_deg']:.0f}",
                        (ox, min(oy + int(rh) + 24, frame.shape[0] - 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if position is not None:
            x_mm, y_mm, z_mm = position[0] * 1000.0, position[1] * 1000.0, position[2] * 1000.0
            text = f'base xyz=({x_mm:.0f},{y_mm:.0f},{z_mm:.0f})mm depth_valid={bool(depth_valid)}'
            cv2.putText(frame, text, (6, frame.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            return
        msg = CompressedImage()
        msg.header = color_msg.header
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self.pub_debug_image.publish(msg)

    def _track_hand(self, color_msg, depth_msg, info_msg, tf_at_stamp):
        """MediaPipe로 매 프레임 손 위치·주먹 여부를 판별해 HandTrack을 만든다.

        ToolTrack과 달리 미검출/복원 실패 프레임도 detected=False로 계속 발행한다 -
        robot_control의 HandServoLoop는 "마지막으로 메시지를 받은 시각"으로 손 유실
        (t_lost_s)을 판정하므로(should_abort와 동일한 설계), 이 노드가 조용히 멈추지
        않고 매 프레임 신호를 보내야 한다. 반환값은 절대 None이 아니다(_safe_call의
        NotImplementedError 경로만 예외)."""
        track = HandTrack()
        track.header = color_msg.header
        track.header.frame_id = 'base_link'
        track.pose.orientation.w = 1.0  # 손 자세는 위치만 쓰고 방향은 무시

        if self._hands_detector is None:
            self._hands_detector = create_hands_detector()

        image = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        landmarks, confidence = detect_hand_landmarks(self._hands_detector, image)
        if landmarks is None:
            self._fist_confirm_count = 0
            self.get_logger().warn('MediaPipe가 손을 찾지 못했습니다.', throttle_duration_sec=1.0)
            track.detected = False
            return track

        h, w = image.shape[:2]
        px, py = int(landmarks[0].x * w), int(landmarks[0].y * h)  # 0번 = 손목(WRIST)
        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if not (0 <= py < depth_image.shape[0] and 0 <= px < depth_image.shape[1]):
            self._fist_confirm_count = 0
            self.get_logger().warn(
                f'손목 픽셀({px},{py})이 depth 이미지 범위를 벗어났습니다. '
                f'(width={depth_image.shape[1]}, height={depth_image.shape[0]})',
                throttle_duration_sec=1.0)
            track.detected = False
            return track
        raw_depth_mm = depth_image[py, px]
        depth_m = float(raw_depth_mm) / 1000.0
        if depth_m <= 0.0:
            self._fist_confirm_count = 0
            self.get_logger().warn(
                f'손목 픽셀({px},{py}) depth가 0 이하입니다. (raw_depth_mm={raw_depth_mm})',
                throttle_duration_sec=1.0)
            track.detected = False
            return track

        fx, fy, ppx, ppy = (float(info_msg.k[0]), float(info_msg.k[4]),
                            float(info_msg.k[2]), float(info_msg.k[5]))
        cam_xyz = pixel_to_camera_xyz(px, py, depth_m, fx, fy, ppx, ppy)
        base_xyz = camera_to_base(cam_xyz, self._tf_matrix(tf_at_stamp))

        raw_fist = is_fist(landmarks)
        self._fist_confirm_count = self._fist_confirm_count + 1 if raw_fist else 0
        fist_confirm_frames = self.get_parameter('vision.fist_confirm_frames').value
        confirmed_fist = self._fist_confirm_count >= fist_confirm_frames

        track.pose.position.x, track.pose.position.y, track.pose.position.z = base_xyz
        track.detected = True
        track.fist = confirmed_fist
        track.confidence = confidence
        self._checkpoint_event(
            'H', 'hand_pose_published', 'PASS',
            'base_link 기준 hand_track을 발행합니다.',
            {
                'px': px, 'py': py, 'depth_m': depth_m,
                'base_xyz': [float(v) for v in base_xyz],
                'raw_fist': raw_fist,
                'fist_confirm_count': self._fist_confirm_count,
                'confirmed_fist': confirmed_fist,
            },
            throttle_s=1.0)
        return track


    def destroy_node(self):
        if self._timing_csv is not None:
            self._timing_csv.close()
            self._timing_csv = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
