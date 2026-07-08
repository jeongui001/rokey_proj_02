import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener, TransformException
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from handover_interfaces.msg import ToolTrack, DetectionArray
from handover_interfaces.srv import SetVisionMode

from vision_node.tracking import (
    ToolTracker, pixel_to_camera_xyz, transform_to_matrix, camera_to_base, is_approaching,
)
from vision_node.hand_tracking import create_hands_detector, detect_hand_wrist_pixel
from vision_node.grasp_geometry import (
    AxisSmoother, is_bbox_at_edge, patch_median_depth, tool_axis_from_depth,
    yaw_deg_to_quaternion,
)

# 뎁스/축 계산 상수 - tool_detection_node(프로토타입 계열)에서 실기 검증된 값 그대로.
DEPTH_MAX_M = 2.0        # 이보다 먼 뎁스는 무효(배경/노이즈)
PATCH_HALF = 4           # patch_median_depth 반경 -> (2*4+1)^2 = 9x9 패치
YAW_MIN_MASK_PX = 50     # 공구 윗면 마스크 최소 픽셀 수 - 미달이면 축 계산 포기
FOV_MARGIN_PX = 8        # bbox가 화면 가장자리에 이만큼 가까우면 잘림 의심 플래그
# 장단축비(길이/폭)가 이 값 이상이면 PCA 각도를 완전히 신뢰. 정사각형에 가까울수록
# (렌치/망치 머리처럼 폭이 넓은 공구) 각도가 노이즈에 민감해 튀므로 신뢰도를 낮춘다.
ELONGATION_TRUST_MIN = 1.3
ELONGATION_ALPHA_FLOOR = 0.2  # 저신뢰 구간에서 스무딩 alpha에 곱할 최소 배율


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
        self.min_z_m = self.get_parameter('vision.min_z_m').value
        self.yaw_band = self.get_parameter('vision.yaw_depth_band_m').value
        self.valid_min_ratio = self.get_parameter('vision.depth_valid_min_ratio').value
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

        self.pub_tool_track = self.create_publisher(ToolTrack, '/vision/tool_track', 10)  # 서브스크라이버: task_manager, robot_control(servo_pick 중 직접 구독)
        self.pub_hand_pose = self.create_publisher(PoseStamped, '/vision/hand_pose', 10)  # 서브스크라이버: task_manager
        self.srv_set_mode = self.create_service(SetVisionMode, '/vision/set_mode', self._on_set_mode)  # 클라이언트: task_manager

        # eye-in-hand라 3D 복원엔 "지금"이 아니라 "이미지가 찍힌 시각"의 flange pose가 필요하다
        # (전체 계획.md 2.4절) - 그래서 TF를 캐시해두고 이미지 stamp로 lookup_transform 한다.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # color/depth/camera_info/검출결과 4개를 같은 시각 근처끼리 묶어서 하나의 콜백으로 받는다.
        # 이래야 "이 bbox가 어느 depth 프레임의 것인지"가 어긋나지 않는다.
        # realsense2_camera는 이미지/camera_info를 BEST_EFFORT(sensor data) QoS로 퍼블리시한다.
        # qos_profile을 안 주면 message_filters.Subscriber가 기본값(RELIABLE)을 써서 DDS가
        # 아예 연결을 안 맺어 콜백이 영원히 안 불린다 - 반드시 맞춰줘야 한다.
        self.sub_color = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw', qos_profile=qos_profile_sensor_data)  # 퍼블리셔: realsense2_camera(기성 패키지)
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

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _on_set_mode(self, request, response):
        """task_manager가 부르는 서비스 핸들러. 모드를 TRACK_TOOL로 새로 켤 때마다
        추적기를 리셋해서 이전 물체(다른 공구, 이전 사이클의 잔상)를 안 물고 가게 한다."""
        self.mode = request.mode
        self.tool_class = request.tool_class
        if request.mode == SetVisionMode.Request.TRACK_TOOL:
            self.tracker.reset()
            self.axis_smoother.reset(request.tool_class)  # 이전 사이클의 축 이력도 함께 초기화
        response.success = True
        response.message = f'mode set to {request.mode} (tool_class={request.tool_class})'
        return response

    def _on_synced_images(self, color_msg, depth_msg, info_msg, detection_msg):
        """4개 토픽이 시간적으로 맞춰졌을 때마다(30~60Hz 목표) 호출되는 메인 루프.
        모드에 따라 _track_tool 또는 _track_hand로 위임하고, 결과가 있으면 퍼블리시."""
        # DEBUG(파이프라인 점검용, 나중에 제거): 동기화 콜백 자체가 도는지 확인
        self.get_logger().info(
            f'DEBUG sync callback fired, mode={self.mode}, tool_class={self.tool_class!r}, '
            f'n_detections={len(detection_msg.detections)}',
            throttle_duration_sec=1.0)
        try:
            # "지금"이 아니라 color_msg가 찍힌 시각의 flange pose로 조회 (2.4절 핵심)
            tf_at_stamp = self.tf_buffer.lookup_transform(
                'base_link', color_msg.header.frame_id, color_msg.header.stamp,
                timeout=Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup failed: {ex}')
            return

        if self.mode == SetVisionMode.Request.TRACK_TOOL:
            track = self._safe_call(
                self._track_tool, color_msg, depth_msg, info_msg, detection_msg,
                tf_at_stamp, self.tool_class, default=None)
            if track is not None:
                self.pub_tool_track.publish(track)
        elif self.mode == SetVisionMode.Request.TRACK_HAND:
            hand_pose = self._safe_call(
                self._track_hand, color_msg, depth_msg, info_msg, tf_at_stamp, default=None)
            if hand_pose is not None:
                self.pub_hand_pose.publish(hand_pose)
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
        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_m_img = depth_image.astype(np.float64) / 1000.0  # RealSense depth는 보통 mm(16UC1)
        tf_matrix = self._tf_matrix(tf_at_stamp)

        def reconstruct(cx, cy):
            """bbox 중심 픽셀(cx, cy) -> base_link 3D 좌표. ToolTracker.update()가
            후보 bbox마다 이 함수를 호출한다 (tracking.py는 ROS/depth 이미지를 몰라도 되게
            이 클로저 하나로 depth 조회 + intrinsics + tf 변환을 전부 감춘다)."""
            px, py = int(cx), int(cy)
            if not (0 <= py < depth_m_img.shape[0] and 0 <= px < depth_m_img.shape[1]):
                return None
            # 단일 픽셀은 금속/반사면 뎁스 구멍에 취약해서 9x9 패치 median으로 보완
            z_m, valid_ratio = patch_median_depth(
                depth_m_img, px, py, half=PATCH_HALF, dmin=self.min_z_m, dmax=DEPTH_MAX_M)
            depth_valid = z_m is not None and valid_ratio >= self.valid_min_ratio
            # depth 무효 구간은 마지막 유효 z로 픽셀->광선을 역산해 x,y만 RGB 추적으로 갱신 (2.7절)
            z = z_m if z_m is not None else (self.tracker.last_valid_z or 0.0)
            cam_xyz = pixel_to_camera_xyz(px, py, z, fx, fy, ppx, ppy)
            base_xyz = camera_to_base(cam_xyz, tf_matrix)
            return (base_xyz[0], base_xyz[1], base_xyz[2], depth_valid)

        stamp = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
        result = self.tracker.update(detection_msg.detections, tool_class, reconstruct, stamp)
        # DEBUG(파이프라인 점검용, 나중에 제거): tracker.update 결과 확인
        self.get_logger().info(
            f'DEBUG _track_tool: classes_in_frame='
            f'{[d.class_name for d in detection_msg.detections]}, '
            f'looking_for={tool_class!r}, result_is_none={result is None}',
            throttle_duration_sec=1.0)
        if result is None:
            # 검출이 끊긴 프레임 - 축 이력도 지운다. 물체가 화면에서 사라졌다 다시
            # 나타났을 때 이전 물체의 각도가 새 물체 각도와 섞이는 것을 막는다
            # (프로토타입 reset_missing과 같은 방침).
            self.axis_smoother.reset(tool_class)
            return None  # 이번 프레임엔 tool_class 검출이 없었음 - 퍼블리시 안 함

        position, velocity, depth_valid, chosen_det = result
        yaw_quat = self._grip_yaw_quaternion(
            chosen_det, depth_m_img, fx, fy, ppx, ppy, tf_matrix, tool_class)

        track = ToolTrack()
        track.header = color_msg.header  # 관측 시각(stamp)은 그대로 - 서보 루프의 시간 정합 기준
        # position/orientation은 위에서 이미 base_link로 변환했으므로 frame_id도 base_link로
        # 고쳐야 한다 - color_msg.header를 그대로 복사하면 frame_id가 카메라 프레임으로 남아
        # robot_control의 _validate_tool_track_message(frame_id=='base_link' 검사)가 거부한다.
        # _track_tool은 TF 조회 성공(_on_synced_images) 후에만 호출되므로 항상 base_link.
        track.header.frame_id = 'base_link'
        track.tool_class = tool_class
        track.confidence = float(chosen_det.score)
        track.pose.position.x = position[0]
        track.pose.position.y = position[1]
        track.pose.position.z = position[2]
        if yaw_quat is not None:
            (track.pose.orientation.x, track.pose.orientation.y,
             track.pose.orientation.z, track.pose.orientation.w) = yaw_quat
        else:
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
        마스크 픽셀 부족 등으로 축을 못 구한 프레임은 None."""
        h, w = depth_m_img.shape
        x1, y1 = max(det.x1, 0), max(det.y1, 0)
        x2, y2 = min(det.x2, w), min(det.y2, h)
        if x2 <= x1 or y2 <= y1:
            return None
        if is_bbox_at_edge(det.x1, det.y1, det.x2, det.y2, w, h, FOV_MARGIN_PX):
            # 가장자리 걸침은 컨베이어 위에서 흔해 무효화하면 축이 거의 안 나온다 -
            # 보이는 부분만으로 계속 계산하되 정보성 로그만 남긴다 (프로토타입 방침).
            self.get_logger().debug(
                f'{det.class_name} bbox가 화면 가장자리에 걸침 - 잘린 실루엣으로 축 계산',
                throttle_duration_sec=1.0)
        roi = depth_m_img[y1:y2, x1:x2]
        axis_deg, _, elongation = tool_axis_from_depth(
            roi, fx, fy, ppx, ppy, ox=x1, oy=y1,
            dmin=self.min_z_m, dmax=DEPTH_MAX_M,
            band_m=self.yaw_band, min_px=YAW_MIN_MASK_PX)
        if axis_deg is None:
            return None
        trust = min(1.0, max(0.0, (elongation - 1.0) / (ELONGATION_TRUST_MIN - 1.0)))
        alpha = self.axis_smoother.alpha * max(ELONGATION_ALPHA_FLOOR, trust)
        axis_deg = self.axis_smoother.update(tool_class, axis_deg, alpha=alpha)
        grip_deg = (axis_deg + 90.0) % 180.0  # top-down 파지: 장축에 수직으로 닫음

        # 그립축 방향벡터(카메라 이미지 평면)를 base 좌표계로 회전 (top-down 전제)
        d_cam = np.array([np.cos(np.deg2rad(grip_deg)), np.sin(np.deg2rad(grip_deg)), 0.0])
        rot = np.array(tf_matrix)[:3, :3]
        d_base = rot @ d_cam
        base_yaw_deg = float(np.degrees(np.arctan2(d_base[1], d_base[0])) % 180.0)
        return yaw_deg_to_quaternion(base_yaw_deg)

    def _track_hand(self, color_msg, depth_msg, info_msg, tf_at_stamp):
        """MediaPipe로 손목을 검출해 PoseStamped를 만든다."""
        if self._hands_detector is None:
            self._hands_detector = create_hands_detector()

        image = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        wrist_px = detect_hand_wrist_pixel(self._hands_detector, image)
        if wrist_px is None:
            return None  # 손 미검출 - task_manager의 hand_detect_timeout_s가 폴백 처리

        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        px, py = wrist_px
        if not (0 <= py < depth_image.shape[0] and 0 <= px < depth_image.shape[1]):
            return None
        depth_m = float(depth_image[py, px]) / 1000.0
        if depth_m <= 0.0:
            return None

        fx, fy, ppx, ppy = (float(info_msg.k[0]), float(info_msg.k[4]),
                            float(info_msg.k[2]), float(info_msg.k[5]))
        cam_xyz = pixel_to_camera_xyz(px, py, depth_m, fx, fy, ppx, ppy)
        base_xyz = camera_to_base(cam_xyz, self._tf_matrix(tf_at_stamp))

        pose = PoseStamped()
        pose.header = color_msg.header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = base_xyz
        pose.pose.orientation.w = 1.0  # 손 자세는 위치만 쓰고 방향은 무시
        return pose


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
