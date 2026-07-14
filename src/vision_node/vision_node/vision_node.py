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
from rclpy.qos import (
    QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data,
)
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Bool, String

from handover_interfaces.msg import ToolTrack, HandTrack, DetectionArray, VisionTiming
from handover_interfaces.srv import SetVisionMode

from vision_node.tracking import (
    ToolTracker, pixel_to_camera_xyz, transform_to_matrix, camera_to_base, is_approaching,
    KPT_CONF_MIN, detection_center,
)
from vision_node.grasp_geometry import (
    AxisSmoother, align_depth_to_color, is_bbox_at_edge, patch_median_depth,
    tool_axis_from_depth, yaw_deg_to_quaternion,
)

# 캘리브레이션이 이미 광학좌표계 기준이라 color_msg.header.frame_id로 조회하면 회전이
# 중복 적용된다(x<->z 섞임) - link_6->광학좌표계를 직접 발행하는 이 프레임을 조회한다.
CAMERA_OPTICAL_CALIB_FRAME = 'camera_optical_calib'

# 뎁스/축 계산 상수 - tool_detection_node에서 검증된 값
DEPTH_MAX_M = 2.0        # 배경/노이즈 컷오프
PATCH_HALF = 2           # patch_median_depth 반경 (9x9 패치)
YAW_MIN_MASK_PX = 50     # 미달 시 축 계산 포기
FOV_MARGIN_PX = 8        # 화면 가장자리 근접 시 잘림 의심
# 장단축비가 이 값 이상이면 PCA 각도를 완전히 신뢰 - 정사각형에 가까울수록(폭 넓은 공구) 노이즈에 민감
ELONGATION_TRUST_MIN = 1.3
ELONGATION_ALPHA_FLOOR = 0.2  # 저신뢰 구간 alpha 최소 배율

# 디버그 이미지용 클래스 색상 팔레트(BGR) - 클래스 목록을 모르는 상태라 이름 해시로 고정 배정
_DEBUG_CLASS_COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
]

# 원본 해상도(424x240)는 확대해야 라벨이 읽기 쉬움 - 연산은 원본, 발행 직전에만 확대
_DEBUG_IMAGE_UPSCALE = 2
_DEBUG_FONT_SCALE = 0.35 * _DEBUG_IMAGE_UPSCALE
_DEBUG_FONT_THICKNESS = 1


def _class_color(class_name):
    return _DEBUG_CLASS_COLORS[hash(class_name) % len(_DEBUG_CLASS_COLORS)]


def _put_text_clamped(frame, text, x, y, color,
                       font_scale=_DEBUG_FONT_SCALE, thickness=_DEBUG_FONT_THICKNESS):
    """cv2.putText 래퍼 - 텍스트 원점을 프레임 경계 안으로 clamp한다."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(0, min(x, w - tw))
    y = max(th, min(y, h - 1))
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


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

        self.declare_parameter('vision.min_z_m', 0.10)         # 이보다 가까우면 depth 무효 취급
        self.declare_parameter('vision.approach_ref_x', 0.0)
        self.declare_parameter('vision.approach_ref_y', 0.0)
        self.declare_parameter('vision.tracker_alpha', 0.6)
        self.declare_parameter('vision.tracker_beta', 0.3)
        # p0/p1/bbox 폴백 모드 전용 z 스무딩 - mid(양쪽 kpt 중점) 대비 depth 노이즈가 커서 더 세게 누른다
        self.declare_parameter('vision.tracker_alpha_z_offset_mode', 0.15)
        # 축(yaw) 계산 파라미터 - tool_detection_node와 동일 기본값
        self.declare_parameter('vision.yaw_depth_band_m', 0.008)  # 공구 윗면에서 이보다 깊은 픽셀은 벨트로 보고 제외
        self.declare_parameter('vision.axis_smooth_alpha', 0.25)
        self.declare_parameter('vision.depth_valid_min_ratio', 0.2)  # 패치 유효 비율이 이 미만이면 depth_valid=False
        # 매 프레임 인코딩 비용이 있어 필요 없으면 끌 수 있게 파라미터화 (operator_gui 구독)
        self.declare_parameter('vision.publish_debug_image', True)
        # 주먹 확정 연속 프레임 수(디바운스) - HandTrack.fist는 이미 확정된 값이라는 계약이라
        # (robot_control의 HandServoLoop가 재확인 없이 정지 처리) 여기서 책임진다.
        self.declare_parameter('vision.fist_confirm_frames', 5)
        # mediapipe 컨테이너 분리로 손 검출 결과가 비동기 도착 - 이 시간(초)보다 오래된 캐시는 미검출로 간주
        self.declare_parameter('vision.hand_detection_max_age_s', 0.2)
        self.declare_parameter('debug.publish_events', True)
        self.declare_parameter('debug.log_vision_decisions', False)
        # FPS 저하 vs latency 누적 판별용 구간 타이밍. csv_path 지정 시 프레임당 CSV도 기록(tools/analyze_timing.py)
        self.declare_parameter('vision.publish_timing', True)
        self.declare_parameter('vision.timing_csv_path', '')
        self.min_z_m = self.get_parameter('vision.min_z_m').value
        self.yaw_band = self.get_parameter('vision.yaw_depth_band_m').value
        self.valid_min_ratio = self.get_parameter('vision.depth_valid_min_ratio').value
        self.publish_debug_image = self.get_parameter('vision.publish_debug_image').value
        self.fist_confirm_frames = self.get_parameter('vision.fist_confirm_frames').value
        self.hand_detection_max_age_s = self.get_parameter('vision.hand_detection_max_age_s').value
        self.approach_ref_xy = (
            self.get_parameter('vision.approach_ref_x').value,
            self.get_parameter('vision.approach_ref_y').value,
        )

        self._bridge = CvBridge()
        self.tracker = ToolTracker(
            alpha=self.get_parameter('vision.tracker_alpha').value,
            beta=self.get_parameter('vision.tracker_beta').value,
            alpha_z_offset_mode=self.get_parameter(
                'vision.tracker_alpha_z_offset_mode').value)
        self.axis_smoother = AxisSmoother(
            alpha=self.get_parameter('vision.axis_smooth_alpha').value)
        self._hand_detection = None  # (payload dict, 수신 시각) - 컨테이너의 최신 검출 캐시
        self._fist_counter = 0  # TRACK_HAND에서 is_fist 연속 참 카운트(디바운스)
        # realsense2_camera의 align_depth 기능이 이 카메라/드라이버 조합에서 깨져 있어(요청 fps의
        # 2배로 나오거나 0프레임 발행), raw depth를 받아 grasp_geometry.align_depth_to_color로 직접 정렬한다.
        self._depth_intrinsics = None  # (fx,fy,ppx,ppy) - /camera/depth/camera_info 캐시
        self._depth_to_color = None  # (rotation 3x3, translation 3,) - 뎁스->컬러 TF 캐시(고정값이라 최초 1회만 조회)

        self.pub_tool_track = self.create_publisher(ToolTrack, '/vision/tool_track', 10)  # 서브스크라이버: task_manager, robot_control(servo_pick 중 직접 구독)
        self.pub_hand_track = self.create_publisher(HandTrack, '/vision/hand_track', 10)  # 서브스크라이버: robot_control(handover_approach 중 직접 구독)
        self.pub_debug_image = self.create_publisher(
            CompressedImage, '/vision/debug_image/compressed', 10)  # 서브스크라이버: operator_gui, 모니터링용(rqt_image_view 등)
        self.pub_debug_events = self.create_publisher(String, '/debug/events', 10)
        self.pub_timing = self.create_publisher(VisionTiming, '/perception/timing', 10)  # 서브스크라이버: 팀 모니터링/tools 분석
        # 컨테이너를 TRACK_HAND 구간에서만 구동시키는 게이트. transient_local로 래치해 컨테이너가
        # 늦게 떠도 현재 모드를 즉시 알 수 있게 함(구독자도 동일 QoS 필요). 컨테이너엔
        # handover_interfaces가 없어 std_msgs 타입만 쓸 수 있다.
        gate_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_hand_enable = self.create_publisher(
            Bool, '/vision/hand_track_enable', gate_qos)  # 서브스크라이버: hand_track_docker_node(컨테이너)
        self._publish_hand_enable(False)  # 기동 직후엔 꺼진 상태로 시작
        self._t = {}                            # 이번 프레임의 구간 타이밍/추적 기록 (콜백마다 리셋)
        self._timing_window = deque(maxlen=100)  # (callback_ms, e2e_ms, infer_ms) rolling 통계용
        self._timing_csv = None                 # timing_csv_path 설정 시 지연 오픈
        self.srv_set_mode = self.create_service(SetVisionMode, '/vision/set_mode', self._on_set_mode)  # 클라이언트: task_manager
        # 컨테이너의 mediapipe 결과를 임시로 String(JSON)으로 받는 다리 - HandTrack 커스텀
        # 메시지로 정리되면 제거 예정.
        self.create_subscription(
            String, '/vision/hand_track_docker', self._on_hand_track_docker, 10)

        # eye-in-hand라 3D 복원엔 "지금"이 아니라 이미지가 찍힌 시각의 flange pose가 필요 -
        # TF를 캐시해두고 이미지 stamp로 lookup_transform 한다.
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
        # raw depth 사용 - 정렬은 직접 계산(_align_depth_msg, 위 driver 버그 참고)
        self.sub_depth = message_filters.Subscriber(
            self, Image, '/camera/depth/image_rect_raw',
            qos_profile=qos_profile_sensor_data)  # 퍼블리셔: realsense2_camera(기성 패키지)
        self.sub_info = message_filters.Subscriber(
            self, CameraInfo, '/camera/color/camera_info',
            qos_profile=qos_profile_sensor_data)  # 퍼블리셔: realsense2_camera(기성 패키지)
        self.sub_detections = message_filters.Subscriber(
            self, DetectionArray, '/detection/tool_boxes')  # 퍼블리셔: object_detection(팀원3)
        # depth intrinsics는 프레임마다 안 바뀌는 고정값이라 sync 그룹 없이 최신값만 캐시(_align_depth_msg가 사용)
        self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self._on_depth_info,
            qos_profile=qos_profile_sensor_data)  # 퍼블리셔: realsense2_camera(기성 패키지)
        # queue_size=카메라fps x 허용 검출지연. 60fps에서 10(166ms분량)은 추론 지연 시
        # 동기화가 영구 실패했음(tool_track/debug_image 모두 끊김) - 120(~2초분량)으로 완화.
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info, self.sub_detections],
            queue_size=120, slop=0.05)
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
        """프레임 타이밍을 /perception/timing으로 발행하고, timing_csv_path 설정 시 CSV에도 기록한다.

        callback_ms(연산 합) 대비 e2e_ms(capture->지금)가 크면 큐 적체(추론이 느린 게 아니라 밀린 것)를 뜻한다."""
        if not bool(self.get_parameter('vision.publish_timing').value):
            return
        callback_ms = (time.perf_counter() - t_entry) * 1000.0
        stamp_s = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
        e2e_ms = (self.get_clock().now().nanoseconds * 1e-9 - stamp_s) * 1000.0
        # capture~콜백진입에서 검출 노드 구간을 뺀 값(토픽 홉+동기화 대기). 음수면 노드간 시계 불일치 신호라 clamp 안 함.
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
        """fn이 예외를 던져도 프로세스 전체가 죽지 않게 막는다 - SingleThreadedExecutor는 콜백
        예외를 스핀 루프에서 그대로 재발생시켜 프로세스를 종료시킨다(런치에 respawn 없음)."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            self.get_logger().error(
                f'{fn.__qualname__} 호출 중 예외가 발생해 이번 프레임을 건너뜁니다: '
                f'{exc!r}', throttle_duration_sec=1.0)
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
            self._fist_counter = 0  # 이전 handover 사이클의 주먹 판정 이력을 지운다
            self._hand_detection = None  # 게이트가 꺼져 있던 동안의 마지막 검출 잔상을 지운다
        # TRACK_HAND일 때만 컨테이너 mediapipe 구동 - CPU를 YOLO와 안 나누게 해 검출 지연 방지
        self._publish_hand_enable(request.mode == SetVisionMode.Request.TRACK_HAND)
        response.success = True
        response.message = f'mode set to {request.mode} (tool_class={request.tool_class})'
        checkpoint = self._SET_MODE_CHECKPOINTS.get(request.mode)
        if checkpoint is not None:
            phase, checkpoint_id = checkpoint
            self._checkpoint_event(
                phase, checkpoint_id, 'PASS' if response.success else 'FAIL',
                response.message, {'mode': request.mode, 'tool_class': request.tool_class})
        return response

    def _publish_hand_enable(self, enabled):
        """컨테이너의 mediapipe 추론 on/off 게이트를 발행한다."""
        msg = Bool()
        msg.data = bool(enabled)
        self.pub_hand_enable.publish(msg)

    def _on_hand_track_docker(self, msg):
        """컨테이너가 보낸 mediapipe 결과(JSON)를 캐싱한다. _track_hand가 사용하며
        오래되면(vision.hand_detection_max_age_s) 무시된다."""
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(
                f'/vision/hand_track_docker 메시지를 JSON으로 파싱할 수 없습니다: {msg.data!r}',
                throttle_duration_sec=1.0)
            return
        self._hand_detection = (payload, self.get_clock().now())

    def _on_depth_info(self, msg):
        """뎁스 카메라 intrinsics 캐시(_align_depth_msg가 사용) - 프레임마다 안 바뀌는
        고정값이라 sync 그룹 없이 최신값만 들고 있는다."""
        self._depth_intrinsics = (
            float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))

    def _get_depth_to_color_extrinsics(self, depth_msg, color_msg):
        """뎁스->컬러 외부파라미터를 TF에서 조회해 캐싱한다 - 카메라 리그 고정값이라 최초 1회만 조회."""
        if self._depth_to_color is not None:
            return self._depth_to_color
        try:
            tf = self.tf_buffer.lookup_transform(
                color_msg.header.frame_id, depth_msg.header.frame_id,
                Time(), timeout=Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(
                f'뎁스->컬러 TF 조회에 실패했습니다(정렬 계산 보류): {ex}',
                throttle_duration_sec=1.0)
            return None
        matrix = self._tf_matrix(tf)
        rotation = np.array([row[:3] for row in matrix[:3]])
        translation = np.array([row[3] for row in matrix[:3]])
        self._depth_to_color = (rotation, translation)
        return self._depth_to_color

    def _align_depth_msg(self, depth_msg, color_msg, info_msg):
        """raw depth를 컬러 픽셀 격자에 정렬해 새 Image msg로 반환한다. TF/intrinsics가 아직
        없으면 None(이후 캐시되면 계속 None일 일 없음)."""
        # 실패 사유(TF 없음/intrinsics 없음/정렬 계산 예외)를 구분해 로깅 - 뭉뚱그리면 원인 특정이 오래 걸림
        extrinsics = self._get_depth_to_color_extrinsics(depth_msg, color_msg)
        if extrinsics is None:
            return None  # 사유는 _get_depth_to_color_extrinsics가 이미 로깅함
        if self._depth_intrinsics is None:
            self.get_logger().warn(
                '뎁스 camera_info(intrinsics)를 아직 받지 못해 이번 프레임 정렬을 건너뜁니다 '
                '(/camera/depth/camera_info가 발행되고 있는지 확인하세요).',
                throttle_duration_sec=1.0)
            return None
        rotation, translation = extrinsics
        depth_fx, depth_fy, depth_ppx, depth_ppy = self._depth_intrinsics
        color_fx, color_fy, color_ppx, color_ppy = (
            float(info_msg.k[0]), float(info_msg.k[4]), float(info_msg.k[2]), float(info_msg.k[5]))
        raw_depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        raw_depth_m = raw_depth.astype(np.float64) / 1000.0  # RealSense depth는 보통 mm(16UC1)
        aligned_m = align_depth_to_color(
            raw_depth_m, depth_fx, depth_fy, depth_ppx, depth_ppy,
            color_fx, color_fy, color_ppx, color_ppy,
            rotation, translation, (color_msg.height, color_msg.width),
            dmin=self.min_z_m, dmax=DEPTH_MAX_M)
        aligned_msg = self._bridge.cv2_to_imgmsg(
            (aligned_m * 1000.0).astype(np.uint16), encoding='16UC1')
        aligned_msg.header = color_msg.header
        return aligned_msg

    def _on_synced_images(self, color_msg, depth_msg, info_msg, detection_msg):
        """4개 토픽이 시간적으로 맞춰졌을 때마다(30~60Hz 목표) 호출되는 메인 루프.
        모드에 따라 _track_tool 또는 _track_hand로 위임하고, 결과가 있으면 퍼블리시."""
        t_entry = time.perf_counter()
        entry_wall_s = self.get_clock().now().nanoseconds * 1e-9
        self._t = {}  # 이번 프레임 타이밍 기록 시작 (_track_tool의 구간들이 여기 쌓인다)
        # camera->base TF는 3D변환에만 필요, debug_image는 픽셀좌표만 써서 무관하다. TF 조회
        # 실패해도 tf_at_stamp=None으로 두고 진행 - 3D변환만 건너뛰고 디버그영상은 계속 발행한다
        # (과거엔 실패 시 프레임 전체를 드롭해 카메라 화면이 멈춘 것처럼 보였음).
        t0 = time.perf_counter()
        try:
            # color_msg가 찍힌 시각의 flange pose로 조회 - color_msg.header.frame_id 대신
            # CAMERA_OPTICAL_CALIB_FRAME 사용(회전 중복 방지, 위 상수 설명 참고)
            tf_at_stamp = self.tf_buffer.lookup_transform(
                'base_link', CAMERA_OPTICAL_CALIB_FRAME, color_msg.header.stamp,
                timeout=Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(
                f'이미지 시각의 camera->base TF 조회에 실패했습니다(이번 프레임 3D 변환만 '
                f'건너뜁니다): {ex}',
                throttle_duration_sec=1.0)
            tf_at_stamp = None
        self._t['tf_ms'] = (time.perf_counter() - t0) * 1000.0

        if self.mode == SetVisionMode.Request.TRACK_TOOL:
            # _align_depth_msg 실패는 부분 열화(3D복원만 실패)로 다뤄야 한다 - 전체 드롭 시
            # 디버그영상 발행도 함께 끊겨 GUI가 카메라 꺼짐으로 오인했었다.
            aligned_depth_msg = None
            if tf_at_stamp is not None:
                aligned_depth_msg = self._safe_call(
                    self._align_depth_msg, depth_msg, color_msg, info_msg, default=None)
            track = None
            if aligned_depth_msg is not None:
                track = self._safe_call(
                    self._track_tool, color_msg, aligned_depth_msg, info_msg, detection_msg,
                    tf_at_stamp, self.tool_class, default=None)
                if track is not None:
                    self.pub_tool_track.publish(track)
            elif self.publish_debug_image:
                self._safe_call(
                    self._publish_debug_image, color_msg, detection_msg.detections, None, None,
                    default=None)
            self._publish_timing(color_msg, detection_msg, t_entry, entry_wall_s,
                                 published=track is not None)
        elif self.mode == SetVisionMode.Request.TRACK_HAND:
            # depth 정렬 실패는 부분 열화로 처리 - _track_hand는 detected=False로 계속
            # 발행해야 HandServoLoop가 손 유실과 노드 응답없음을 구분할 수 있다.
            aligned_depth_msg = None
            if tf_at_stamp is not None:
                aligned_depth_msg = self._safe_call(
                    self._align_depth_msg, depth_msg, color_msg, info_msg, default=None)
            hand_track = self._safe_call(
                self._track_hand, color_msg, aligned_depth_msg, info_msg, tf_at_stamp, default=None)
            if hand_track is not None:
                self.pub_hand_track.publish(hand_track)
            # TRACK_HAND 구간에도 디버그영상은 계속 발행 - 안 그러면 GUI가 카메라 정지로 오인
            if self.publish_debug_image:
                self._safe_call(self._publish_debug_image, color_msg, [], None, None, default=None)
        elif self.publish_debug_image:
            # OFF: 추적 데이터는 발행 안 하지만 GUI가 카메라 연결을 확인할 수 있게
            # 원본 화면은 계속 발행(detections=[])
            self._safe_call(self._publish_debug_image, color_msg, [], None, None, default=None)

    def _tf_matrix(self, tf_at_stamp):
        """TransformStamped -> tracking.transform_to_matrix가 쓰는 (translation, rotation) 형태로."""
        t = tf_at_stamp.transform.translation
        r = tf_at_stamp.transform.rotation
        return transform_to_matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    def _track_tool(self, color_msg, depth_msg, info_msg, detection_msg, tf_at_stamp, tool_class):
        """검출 + 3D복원(tf_at_stamp 사용) + 알파-베타 필터로 ToolTrack을 만든다. yaw는 bbox
        depth ROI의 3D PCA로 구한다(grasp_geometry.tool_axis_from_depth)."""
        # CameraInfo.k = 3x3 intrinsic 1차원화: [fx,0,ppx,0,fy,ppy,0,0,1]. float() 캐스팅 안 하면
        # numpy 타입이 섞여 bool 필드(approaching 등) 대입 시 에러.
        fx, fy, ppx, ppy = (float(info_msg.k[0]), float(info_msg.k[4]),
                            float(info_msg.k[2]), float(info_msg.k[5]))
        t0 = time.perf_counter()
        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_m_img = depth_image.astype(np.float64) / 1000.0  # RealSense depth는 보통 mm(16UC1)
        self._t['depth_ms'] = (time.perf_counter() - t0) * 1000.0
        tf_matrix = self._tf_matrix(tf_at_stamp)

        def reconstruct(cx, cy, bbox_w, bbox_h):
            """bbox 중심 픽셀 -> base_link 3D 좌표. tracking.py가 ROS/depth를 몰라도 되도록
            이 클로저에 depth조회+intrinsics+tf변환을 감춘다."""
            px, py = int(cx), int(cy)
            if not (0 <= py < depth_m_img.shape[0] and 0 <= px < depth_m_img.shape[1]):
                self.get_logger().warn(
                    f'bbox 중심 픽셀({px},{py})이 depth 이미지 범위를 벗어났습니다.',
                    throttle_duration_sec=1.0)
                return None
            # 단일 픽셀은 depth 구멍에 취약해 patch median 사용 - bbox가 patch보다 작으면
            # 반경을 bbox 안쪽으로 제한(배경 유입 방지)
            half = max(1, min(PATCH_HALF, int(min(bbox_w, bbox_h) // 2)))
            z_m, valid_ratio = patch_median_depth(
                depth_m_img, px, py, half=half, dmin=self.min_z_m, dmax=DEPTH_MAX_M)
            depth_valid = z_m is not None and valid_ratio >= self.valid_min_ratio
            # depth 무효 시 마지막 유효 z로 x,y만 갱신. z를 지어내면(0.0 등) 엉뚱한 좌표로
            # 로봇이 이동할 위험이 있어 last_valid_z도 없으면 이 후보를 버린다.
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
            # 검출 끊긴 프레임 - 축 이력도 리셋해 다른 물체 각도와 안 섞이게 함
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
        if bool(self.get_parameter('debug.log_vision_decisions').value):
            self.get_logger().info(
                f"tool_track 갱신: mode={self.tracker.last_mode} "
                f"depth_valid={depth_valid} position_m={position}",
                throttle_duration_sec=0.5)
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
        # frame_id를 base_link로 교정 - color_msg 그대로 두면 카메라 프레임으로 남아
        # robot_control의 검증(frame_id=='base_link')이 거부한다.
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
            track.yaw_valid = True
        else:
            self.get_logger().warn(
                f"'{tool_class}' 공구 yaw 축 계산이 불가능해 identity orientation을 사용합니다.",
                throttle_duration_sec=1.0)
            track.pose.orientation.w = 1.0  # 축 미확정 프레임 - identity (구독측은 yaw_valid로 구분)
            track.yaw_valid = False
        track.depth_valid = bool(depth_valid)
        track.approaching = bool(is_approaching(
            (position[0], position[1]), velocity, self.approach_ref_xy))
        return track

    def _grip_yaw_quaternion(self, det, depth_m_img, fx, fy, ppx, ppy, tf_matrix, tool_class):
        """선택된 검출의 bbox depth ROI에서 그립 yaw 쿼터니언(base 기준)을 계산한다.

        장축은 3D PCA로 구하고, 장단축비가 낮을수록(정사각형에 가까울수록) 스무딩을 강하게
        눌러 저신뢰 관측의 영향을 줄인다. 그립 방향은 장축에 수직(top-down). 축을 못 구하면
        (None, None). debug_info는 카메라 이미지 평면 좌표계 그대로라 _publish_debug_image에서
        바로 그릴 수 있다."""
        # pose 모델: keypoint가 있으면 그 벡터각이 곧 장축 - depth-PCA보다 근접/가림에 강건.
        # 없거나 저신뢰면 depth-PCA로 폴백(box 모델 하위호환).
        kpt_axis = self._axis_from_keypoints(det)
        if kpt_axis is not None:
            axis_deg, kpt_debug, trust = kpt_axis
            alpha = self.axis_smoother.alpha * max(ELONGATION_ALPHA_FLOOR, trust)
            axis_deg = self.axis_smoother.update(tool_class, axis_deg, alpha=alpha)
            grip_deg = (axis_deg + 90.0) % 180.0  # top-down 파지: 장축에 수직으로 닫음
            debug_info = {'kpts': kpt_debug, 'axis_deg': axis_deg, 'grip_deg': grip_deg}
            return self._grip_deg_to_base_quaternion(grip_deg, tf_matrix), debug_info
        # 한쪽 kpt만 유효(근접 시 반대쪽이 화면 밖으로 잘림)하고 직전 keypoint 이력이
        # 있으면, 잘린 bbox의 depth-PCA보다 직전 각도를 유지하는 게 정확하다(공구는
        # 접근 중 정지 상태).
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
            # 가장자리 걸침은 컨베이어에서 흔해 무효화하면 축이 거의 안 나옴 - 보이는 부분만으로 계산
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
        """검출의 2-keypoint에서 (axis_deg, 디버그좌표, trust)를 구한다. keypoint 없음/저신뢰/
        두 점이 너무 가까우면 None(depth-PCA로 폴백). trust는 두 kpt conf 평균으로
        AxisSmoother alpha를 누른다."""
        c0, c1 = det.kpt0_conf, det.kpt1_conf
        if c0 < KPT_CONF_MIN or c1 < KPT_CONF_MIN:
            return None
        dx, dy = det.kpt1_x - det.kpt0_x, det.kpt1_y - det.kpt0_y
        # 두 점이 bbox 장변의 30% 미만으로 붙어있으면 각도가 노이즈(끝점 간 거리는 장변 근처가 정상)
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
        """검출 bbox + 추적 대상 축선을 그려 압축 이미지로 발행한다(operator_gui/rqt_image_view
        모니터링용). position/depth_valid가 있으면 계산된 base_link 좌표도 화면에 표시해
        좌표 계산 결과가 말이 되는지 바로 확인할 수 있게 한다."""
        # 원본(424x240) 그대로 그리면 근접 촬영 시 글자가 겹치거나 잘려 upscale해서 그리고
        # _put_text_clamped로 경계 안에 고정한다.
        u = _DEBUG_IMAGE_UPSCALE
        frame = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8').copy()
        frame = cv2.resize(frame, None, fx=u, fy=u, interpolation=cv2.INTER_LINEAR)
        for d in detections:
            color = _class_color(d.class_name)
            is_target = chosen_det is not None and d is chosen_det
            thickness = 3 if is_target else 1
            cv2.rectangle(frame, (d.x1 * u, d.y1 * u), (d.x2 * u, d.y2 * u), color, thickness)
            _put_text_clamped(frame, f'{d.class_name} {d.score * 100:.0f}%',
                               d.x1 * u, max(d.y1 * u - 4, 10), color)

        if axis_debug is not None and 'kpts' in axis_debug:
            # keypoint 경로: 끝점 2개(p0 빨강/p1 파랑)와 그 축선을 그대로 그린다
            (p0x, p0y), (p1x, p1y) = axis_debug['kpts']
            p0 = (int(p0x * u), int(p0y * u))
            p1 = (int(p1x * u), int(p1y * u))
            cv2.line(frame, p0, p1, (255, 255, 255), 1)
            cv2.circle(frame, p0, 3, (0, 0, 255), -1)
            cv2.circle(frame, p1, 3, (255, 0, 0), -1)
            _put_text_clamped(
                frame, f"axis {axis_debug['axis_deg']:.0f} grip {axis_debug['grip_deg']:.0f}",
                min(p0[0], p1[0]), max(p0[1], p1[1]) + 24, (255, 255, 255))
        elif axis_debug is not None and axis_debug.get('held'):
            # 단일 kpt hold 경로: 그릴 기하가 없어 유지 중인 각도만 텍스트로 표시
            _put_text_clamped(
                frame, f"axis {axis_debug['axis_deg']:.0f} grip {axis_debug['grip_deg']:.0f} (hold)",
                6, 16, (255, 255, 255))
        elif axis_debug is not None:
            ox, oy = axis_debug['origin']
            rect = axis_debug['rect']
            (rcx, rcy), (rw, rh), _ = rect
            box_pts = (u * (cv2.boxPoints(rect) + np.array([ox, oy], dtype=np.float32))).astype(np.int32)
            cv2.polylines(frame, [box_pts], True, (255, 255, 255), 1)
            theta = np.deg2rad(axis_debug['axis_deg'])
            half_len = max(rw, rh) / 2
            gcx, gcy = (int(rcx) + ox) * u, (int(rcy) + oy) * u
            dx, dy = int(half_len * u * np.cos(theta)), int(half_len * u * np.sin(theta))
            cv2.line(frame, (gcx - dx, gcy - dy), (gcx + dx, gcy + dy), (255, 255, 255), 1)
            # bbox 위 라벨(클래스명)과 겹치지 않게 회전사각형 하단에 표시
            _put_text_clamped(
                frame, f"axis {axis_debug['axis_deg']:.0f} grip {axis_debug['grip_deg']:.0f}",
                ox * u, (oy + int(rh)) * u + 24, (255, 255, 255))

        if position is not None:
            x_mm, y_mm, z_mm = position[0] * 1000.0, position[1] * 1000.0, position[2] * 1000.0
            text = f'base xyz=({x_mm:.0f},{y_mm:.0f},{z_mm:.0f})mm depth_valid={bool(depth_valid)}'
            _put_text_clamped(frame, text, 6, frame.shape[0] - 6, (0, 255, 0))

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

        ToolTrack과 달리 미검출/실패 프레임도 detected=False로 계속 발행한다 - HandServoLoop가
        마지막 수신 시각으로 손 유실을 판정하므로 조용히 멈추면 안 된다. 반환값은 절대 None이 아니다."""
        track = HandTrack()
        track.header = color_msg.header
        track.header.frame_id = 'base_link'
        track.pose.orientation.w = 1.0  # 손 자세는 위치만 쓰고 방향은 무시

        # 컨테이너가 비동기로 보낸 결과를 캐시에서 사용(이 프레임 정확히가 아니라 충분히 최근 값)
        if self._hand_detection is None:
            self._fist_counter = 0
            self.get_logger().warn(
                '컨테이너에서 손 검출 결과를 아직 받지 못했습니다.', throttle_duration_sec=1.0)
            track.detected = False
            track.fist = False
            track.confidence = 0.0
            return track

        payload, received_at = self._hand_detection
        age_s = (self.get_clock().now() - received_at).nanoseconds * 1e-9
        if age_s > self.hand_detection_max_age_s or not payload.get('detected'):
            self._fist_counter = 0
            self.get_logger().warn(
                f'손 검출 결과가 없거나 오래됐습니다(age_s={age_s:.3f}).',
                throttle_duration_sec=1.0)
            track.detected = False
            track.fist = False
            track.confidence = 0.0
            return track

        px, py = payload['palm_px']
        confidence = payload['confidence']
        track.confidence = float(confidence)

        raw_fist = bool(payload['is_fist'])
        self._fist_counter = self._fist_counter + 1 if raw_fist else 0
        confirmed_fist = self._fist_counter >= self.fist_confirm_frames
        track.fist = confirmed_fist

        if depth_msg is None:
            # 정렬 실패해도 2D 검출/주먹 판정은 유효하니 3D만 스킵하고 그대로 발행(위 docstring의 "절대 None 아님" 유지)
            self.get_logger().warn(
                '뎁스 정렬 결과가 없어 손 3D 위치를 계산할 수 없습니다.',
                throttle_duration_sec=1.0)
            track.detected = False
            return track

        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_m_img = depth_image.astype(np.float64) / 1000.0
        if not (0 <= py < depth_m_img.shape[0] and 0 <= px < depth_m_img.shape[1]):
            self.get_logger().warn(
                f'손바닥 픽셀({px},{py})이 depth 이미지 범위를 벗어났습니다. '
                f'(width={depth_m_img.shape[1]}, height={depth_m_img.shape[0]})',
                throttle_duration_sec=1.0)
            track.detected = False
            return track
        depth_m, valid_ratio = patch_median_depth(
            depth_m_img, px, py, half=PATCH_HALF, dmin=self.min_z_m, dmax=DEPTH_MAX_M)
        if depth_m is None or valid_ratio < self.valid_min_ratio:
            self.get_logger().warn(
                f'손바닥 픽셀({px},{py}) depth가 무효입니다. '
                f'(depth_m={depth_m}, valid_ratio={valid_ratio:.2f})',
                throttle_duration_sec=1.0)
            track.detected = False
            return track

        fx, fy, ppx, ppy = (float(info_msg.k[0]), float(info_msg.k[4]),
                            float(info_msg.k[2]), float(info_msg.k[5]))
        cam_xyz = pixel_to_camera_xyz(px, py, depth_m, fx, fy, ppx, ppy)
        base_xyz = camera_to_base(cam_xyz, self._tf_matrix(tf_at_stamp))

        track.pose.position.x, track.pose.position.y, track.pose.position.z = base_xyz
        track.detected = True
        self._checkpoint_event(
            'H', 'hand_pose_published', 'PASS',
            'base_link 기준 hand_track을 발행합니다.',
            {
                'px': px, 'py': py, 'depth_m': depth_m,
                'base_xyz': [float(v) for v in base_xyz],
                'raw_fist': raw_fist,
                'fist_confirm_count': self._fist_counter,
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
