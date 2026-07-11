import json

import rclpy
import pytest
from rclpy.parameter import Parameter

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from handover_interfaces.msg import ToolTrack, HandTrack, DetectionArray, Detection2D
from handover_interfaces.srv import SetVisionMode
from vision_node.vision_node import VisionNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = VisionNode()
    yield n
    n.destroy_node()


def _make_image_msg():
    msg = Image()
    msg.header = Header()
    msg.header.frame_id = 'camera_link'
    return msg


def _make_info_msg():
    msg = CameraInfo()
    msg.k = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
    return msg


def _make_detection_msg(detections=None):
    msg = DetectionArray()
    msg.detections = detections or []
    return msg


def _fake_bridge_fn(fake_depth):
    """인코딩 인지 imgmsg_to_cv2 대체: 'bgr8' 요청(디버그 이미지)엔 8비트 컬러,
    그 외(depth)엔 fake_depth. depth 배열을 컬러 요청에 그대로 돌려주면
    OpenCV 5.x putText가 CV_8U 검사로 거부한다(4.x는 조용히 통과했음)."""
    import numpy as np

    def fake_imgmsg_to_cv2(msg, desired_encoding=None):
        if desired_encoding == 'bgr8':
            return np.zeros((*fake_depth.shape, 3), dtype=np.uint8)
        return fake_depth
    return fake_imgmsg_to_cv2


def test_set_mode_updates_state(node):
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_TOOL
    request.tool_class = 'spanner'
    response = SetVisionMode.Response()

    result = node._on_set_mode(request, response)

    assert result.success is True
    assert node.mode == SetVisionMode.Request.TRACK_TOOL
    assert node.tool_class == 'spanner'


def test_set_mode_off(node):
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.OFF
    request.tool_class = ''
    response = SetVisionMode.Response()

    result = node._on_set_mode(request, response)

    assert result.success is True
    assert node.mode == SetVisionMode.Request.OFF


def test_synced_images_dispatches_to_track_tool_and_publishes(node):
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tool_class = 'spanner'
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'

    expected_track = ToolTrack()
    expected_track.tool_class = 'spanner'
    node._track_tool = lambda color, depth, info, detection, tf, tool_class: expected_track

    published = []
    node.pub_tool_track.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert published == [expected_track]


def test_synced_images_skips_publish_when_track_tool_returns_none(node):
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._track_tool = lambda *a, **k: None

    published = []
    node.pub_tool_track.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert published == []


def test_synced_images_skips_when_tf_lookup_fails(node):
    from tf2_ros import TransformException

    def _raise(*a, **k):
        raise TransformException('no tf yet')

    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = _raise

    called = []
    node._track_tool = lambda *a, **k: called.append(1)

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert called == []


def test_synced_images_dispatches_to_track_hand(node):
    node.mode = SetVisionMode.Request.TRACK_HAND
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'

    expected_track = HandTrack()
    expected_track.detected = True
    node._track_hand = lambda color, depth, info, tf: expected_track

    published = []
    node.pub_hand_track.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert published == [expected_track]


class FakeTransform:
    class transform:
        class translation:
            x = 0.0
            y = 0.0
            z = 0.0
        class rotation:
            x = 0.0
            y = 0.0
            z = 0.0
            w = 1.0


def test_track_tool_filters_by_class_and_reconstructs_position(node):
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    color_msg = _make_image_msg()
    depth_msg = _make_image_msg()
    info_msg = _make_info_msg()
    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 310, 230, 330, 250
    detection_msg = _make_detection_msg([detection])

    import numpy as np
    fake_depth = np.full((480, 424), 500, dtype=np.uint16)  # 0.5m
    node._bridge.imgmsg_to_cv2 = _fake_bridge_fn(fake_depth)

    track = node._track_tool(
        color_msg, depth_msg, info_msg, detection_msg, FakeTransform(), 'spanner')

    assert track is not None
    assert track.tool_class == 'spanner'
    assert track.pose.position.z == pytest.approx(0.5, abs=1e-3)
    # keypoint 없는 Detection2D는 bbox 모드로 잡히고, bbox는 mid가 아니라서
    # depth_valid가 False로 강제된다(2026-07-11 - mid가 아닌 z는 신뢰하지 않음).
    assert track.depth_valid is False
    assert track.confidence == pytest.approx(0.9, abs=1e-6)
    # position/orientation을 base_link로 변환했으므로 frame_id도 base_link여야 한다 -
    # color_msg.header(카메라 프레임)를 그대로 복사하면 robot_control의 프레임 검증에서 거부됨.
    assert track.header.frame_id == 'base_link'


def test_track_tool_fills_grip_yaw_orientation_from_depth_axis(node):
    """벨트(0.31m) 위에 가로로 놓인 막대(0.30m) 합성 뎁스 - 장축 0도 -> 그립 90도가
    orientation에 반영돼야 한다(기존 뼈대는 identity 고정이었음)."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.full((480, 424), 310, dtype=np.uint16)  # 벨트 0.31m
    fake_depth[100:120, 100:240] = 300                      # 막대 0.30m, 140x20 (장축=x)
    node._bridge.imgmsg_to_cv2 = _fake_bridge_fn(fake_depth)

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 90, 90, 250, 130
    detection_msg = _make_detection_msg([detection])

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is not None
    # 장축 0도(이미지 x축) -> 그립 90도 -> (identity TF라 base에서도 90도) 쿼터니언
    assert track.pose.orientation.z == pytest.approx(np.sin(np.pi / 4), abs=0.05)
    assert track.pose.orientation.w == pytest.approx(np.cos(np.pi / 4), abs=0.05)


def test_track_tool_grip_yaw_from_keypoints_without_depth_axis(node):
    """pose 모델 검출(keypoint 장축 0도)이면 depth 마스크 없이도(평평한 뎁스)
    keypoint 벡터로 grip 90도가 나와야 한다 - depth-PCA 경로를 타지 않는다."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.full((480, 424), 500, dtype=np.uint16)  # 평평한 0.5m - PCA 축 불가 상황
    node._bridge.imgmsg_to_cv2 = _fake_bridge_fn(fake_depth)

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 90, 90, 250, 130
    detection.kpt0_x, detection.kpt0_y, detection.kpt0_conf = 100.0, 110.0, 0.9
    detection.kpt1_x, detection.kpt1_y, detection.kpt1_conf = 240.0, 110.0, 0.9
    detection_msg = _make_detection_msg([detection])

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is not None
    # keypoint 축 0도(이미지 x축) -> 그립 90도 -> (identity TF) 쿼터니언
    assert track.pose.orientation.z == pytest.approx(np.sin(np.pi / 4), abs=0.05)
    assert track.pose.orientation.w == pytest.approx(np.cos(np.pi / 4), abs=0.05)
    assert track.kpt0_x == pytest.approx(100.0)
    assert track.kpt0_y == pytest.approx(110.0)
    assert track.kpt0_conf == pytest.approx(0.9)
    assert track.kpt1_x == pytest.approx(240.0)
    assert track.kpt1_y == pytest.approx(110.0)
    assert track.kpt1_conf == pytest.approx(0.9)


def test_track_tool_position_uses_keypoint_midpoint(node):
    """3D 복원 기준점이 bbox 중심이 아니라 keypoint 중점(파지점)이어야 한다 -
    근접 시 bbox가 붕괴해도 파지점 좌표가 흔들리지 않게 하는 pose 전환의 핵심."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.full((480, 424), 500, dtype=np.uint16)  # 0.5m
    node._bridge.imgmsg_to_cv2 = _fake_bridge_fn(fake_depth)

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 300, 220, 340, 260  # bbox 중심 (320,240)
    detection.kpt0_x, detection.kpt0_y, detection.kpt0_conf = 350.0, 240.0, 0.9
    detection.kpt1_x, detection.kpt1_y, detection.kpt1_conf = 370.0, 240.0, 0.9  # 중점 (360,240)
    detection_msg = _make_detection_msg([detection])

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is not None
    # 중점 (360,240), ppx=320: x = (360-320)*0.5/600 (bbox 중심이면 0.0이 나와버림)
    assert track.pose.position.x == pytest.approx((360 - 320) * 0.5 / 600, abs=1e-3)


def test_track_tool_single_kpt_holds_previous_axis(node):
    """근접으로 한쪽 kpt만 남으면(반대쪽 conf~0) 잘린 ROI의 depth-PCA 대신 직전
    keypoint 축을 유지해야 한다 - 라이브런에서 관측된 yaw 드리프트의 수정."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.full((480, 424), 500, dtype=np.uint16)  # 평평한 뎁스 - PCA 불가
    node._bridge.imgmsg_to_cv2 = _fake_bridge_fn(fake_depth)

    def make_det(kpt1_conf):
        d = Detection2D()
        d.class_name = 'spanner'
        d.score = 0.9
        d.x1, d.y1, d.x2, d.y2 = 90, 90, 250, 130
        d.kpt0_x, d.kpt0_y, d.kpt0_conf = 100.0, 110.0, 0.9
        d.kpt1_x, d.kpt1_y, d.kpt1_conf = 240.0, 110.0, kpt1_conf
        return d

    # 1프레임: 두 kpt 유효 - 축 0도가 스무더에 기록됨
    node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(),
        _make_detection_msg([make_det(0.9)]), FakeTransform(), 'spanner')
    # 2프레임: p1 잘림(conf~0) + 평평한 뎁스라 PCA도 불가 - 직전 축(0도) 유지 기대
    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(),
        _make_detection_msg([make_det(0.002)]), FakeTransform(), 'spanner')

    assert track is not None
    # 유지된 축 0도 -> 그립 90도 쿼터니언 (identity가 아니어야 한다 - 수정 전엔
    # PCA 실패로 orientation identity 폴백이었음)
    assert track.pose.orientation.z == pytest.approx(np.sin(np.pi / 4), abs=0.05)
    assert track.pose.orientation.w == pytest.approx(np.cos(np.pi / 4), abs=0.05)


def test_track_tool_kpt_axis_too_short_falls_back_to_depth_axis(node):
    """keypoint가 뭉쳐 있으면(축 길이 < bbox 장변 30%) 각도를 신뢰하지 않고
    기존 depth-PCA 경로로 폴백해야 한다."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.full((480, 424), 310, dtype=np.uint16)  # 벨트 0.31m
    fake_depth[100:120, 100:240] = 300                      # 막대 0.30m (장축=x)
    node._bridge.imgmsg_to_cv2 = _fake_bridge_fn(fake_depth)

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 90, 90, 250, 130
    # 두 점이 3px 간격 - 장변 160px의 30%(48px) 미만이라 무시돼야 함
    detection.kpt0_x, detection.kpt0_y, detection.kpt0_conf = 168.0, 110.0, 0.9
    detection.kpt1_x, detection.kpt1_y, detection.kpt1_conf = 171.0, 110.0, 0.9
    detection_msg = _make_detection_msg([detection])

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is not None
    # depth-PCA 폴백으로도 장축 0도 -> 그립 90도가 나와야 한다
    assert track.pose.orientation.z == pytest.approx(np.sin(np.pi / 4), abs=0.05)
    assert track.pose.orientation.w == pytest.approx(np.cos(np.pi / 4), abs=0.05)


def test_synced_images_publishes_timing_in_track_tool_mode(node):
    """TRACK_TOOL 콜백마다 /perception/timing으로 VisionTiming이 나가야 한다 -
    구간 타이밍(infer/sync_wait/e2e)과 published 플래그(miss rate용) 포함."""
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tool_class = 'spanner'
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._track_tool = lambda *a, **k: None  # 검출 miss 프레임

    timing = []
    node.pub_timing.publish = timing.append

    detection_msg = _make_detection_msg()
    detection_msg.infer_ms = 12.5
    detection_msg.detect_latency_ms = 20.0
    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg)

    assert len(timing) == 1
    t = timing[0]
    assert t.infer_ms == pytest.approx(12.5)
    assert t.published == 0          # track None -> miss 프레임
    assert t.callback_ms >= 0.0
    assert t.e2e_ms != 0.0           # capture stamp(0) 대비 현재 시각 - 0일 수 없다


def test_synced_images_no_timing_when_disabled(node):
    """vision.publish_timing=False면 타이밍 발행을 완전히 건너뛴다."""
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._track_tool = lambda *a, **k: None
    node.set_parameters([Parameter('vision.publish_timing', value=False)])

    timing = []
    node.pub_timing.publish = timing.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert timing == []


def test_track_tool_orientation_identity_when_axis_unavailable(node):
    """축을 못 구하는 프레임은 orientation이 identity로 남아야 한다.

    주의: '전체 depth 무효 + 첫 프레임'으로 만들면 안 된다 - 그 경우 z=0 폴백 가드
    (2026-07-08 실기 사고 수정)가 track 자체를 안 만드는 게 옳은 동작이다. 여기선
    depth는 유효하되 bbox가 너무 작아(마스크 < YAW_MIN_MASK_PX) 축만 불가한 상황."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.full((480, 424), 500, dtype=np.uint16)  # 유효 depth 0.5m
    node._bridge.imgmsg_to_cv2 = _fake_bridge_fn(fake_depth)

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 300, 230, 306, 236  # 6x6=36px < 50
    detection_msg = _make_detection_msg([detection])

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is not None
    assert track.pose.orientation.w == pytest.approx(1.0)
    assert track.pose.orientation.z == pytest.approx(0.0)


def test_track_tool_publishes_debug_image_with_bbox_and_axis_overlay(node):
    """전체 계획.md 4.6절 계약: bbox+축 오버레이를 /vision/debug_image/compressed로
    발행해야 한다 - operator_gui/rqt_image_view 모니터링용(기존엔 미구현이었음)."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.full((480, 424), 310, dtype=np.uint16)  # 벨트 0.31m
    fake_depth[100:120, 100:240] = 300                      # 막대 0.30m, 140x20

    def fake_imgmsg_to_cv2(msg, desired_encoding=None):
        if desired_encoding == 'bgr8':
            return np.zeros((480, 424, 3), dtype=np.uint8)
        return fake_depth

    node._bridge.imgmsg_to_cv2 = fake_imgmsg_to_cv2

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 90, 90, 250, 130
    detection_msg = _make_detection_msg([detection])

    published = []
    node.pub_debug_image.publish = published.append

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is not None
    assert len(published) == 1
    assert published[0].format == 'jpeg'
    assert len(published[0].data) > 0


def test_track_tool_debug_image_off_when_disabled(node):
    """vision.publish_debug_image=False면 인코딩 비용 없이 발행을 건너뛴다."""
    node.publish_debug_image = False
    node.tf_buffer.lookup_transform = lambda *a, **k: None
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: __import__('numpy').zeros(
        (480, 424), dtype='uint16')

    detection_msg = _make_detection_msg([])
    published = []
    node.pub_debug_image.publish = published.append

    node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert published == []


# ---- _track_hand (HandTrack 연속 발행 + 주먹 판별) ----

class FakeHandLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


_FINGER_JOINT_GROUPS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def _make_hand_landmarks(wrist_x=0.5, wrist_y=0.5, fist=True):
    """WRIST(0)을 (wrist_x, wrist_y)에 두고, 4개 손가락을 fist 여부에 맞게(pip/tip
    거리 관계) 배치한 21개짜리 랜드마크 리스트를 만든다."""
    landmarks = [FakeHandLandmark(wrist_x, wrist_y) for _ in range(21)]
    pip_d, tip_d = (0.05, 0.02) if fist else (0.05, 0.10)
    for mcp_i, pip_i, dip_i, tip_i in _FINGER_JOINT_GROUPS:
        landmarks[mcp_i] = FakeHandLandmark(wrist_x + pip_d * 0.5, wrist_y)
        landmarks[pip_i] = FakeHandLandmark(wrist_x + pip_d, wrist_y)
        landmarks[dip_i] = FakeHandLandmark(wrist_x + (pip_d + tip_d) / 2, wrist_y)
        landmarks[tip_i] = FakeHandLandmark(wrist_x + tip_d, wrist_y)
    return landmarks


def _patch_hand_landmarks(node, monkeypatch, landmarks, confidence=0.9):
    import vision_node.vision_node as vision_node_module
    node._hands_detector = object()  # create_hands_detector()의 실제 mediapipe 호출을 피한다
    monkeypatch.setattr(
        vision_node_module, 'detect_hand_landmarks',
        lambda detector, image: (landmarks, confidence))


def _setup_hand_depth(node, depth_mm=500):
    import numpy as np

    def fake_imgmsg_to_cv2(msg, desired_encoding=None):
        if desired_encoding == 'bgr8':
            return np.zeros((480, 424, 3), dtype=np.uint8)
        return np.full((480, 424), depth_mm, dtype=np.uint16)

    node._bridge.imgmsg_to_cv2 = fake_imgmsg_to_cv2


def test_track_hand_returns_not_detected_when_no_hand_found(node, monkeypatch):
    _patch_hand_landmarks(node, monkeypatch, None, 0.0)
    _setup_hand_depth(node)

    track = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), FakeTransform())

    assert track is not None  # ToolTrack과 달리 미검출도 계속 발행(detected=False)
    assert track.detected is False
    assert track.header.frame_id == 'base_link'


def test_track_hand_resets_fist_confirm_count_when_hand_missing(node, monkeypatch):
    _setup_hand_depth(node)
    _patch_hand_landmarks(node, monkeypatch, _make_hand_landmarks(fist=True))
    node._track_hand(_make_image_msg(), _make_image_msg(), _make_info_msg(), FakeTransform())
    assert node._fist_confirm_count == 1

    _patch_hand_landmarks(node, monkeypatch, None, 0.0)
    node._track_hand(_make_image_msg(), _make_image_msg(), _make_info_msg(), FakeTransform())

    assert node._fist_confirm_count == 0


def test_track_hand_reconstructs_position_when_detected(node, monkeypatch):
    _setup_hand_depth(node, depth_mm=500)
    _patch_hand_landmarks(node, monkeypatch, _make_hand_landmarks(wrist_x=0.5, wrist_y=0.5, fist=False))

    track = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), FakeTransform())

    assert track.detected is True
    assert track.pose.position.z == pytest.approx(0.5, abs=1e-3)
    assert track.fist is False


def test_track_hand_open_hand_never_sets_fist(node, monkeypatch):
    _setup_hand_depth(node)
    node.set_parameters([Parameter('vision.fist_confirm_frames', value=3)])
    _patch_hand_landmarks(node, monkeypatch, _make_hand_landmarks(fist=False))

    for _ in range(5):
        track = node._track_hand(
            _make_image_msg(), _make_image_msg(), _make_info_msg(), FakeTransform())

    assert track.fist is False


def test_track_hand_confirms_fist_only_after_n_consecutive_frames(node, monkeypatch):
    node.set_parameters([Parameter('vision.fist_confirm_frames', value=3)])
    _setup_hand_depth(node)
    _patch_hand_landmarks(node, monkeypatch, _make_hand_landmarks(fist=True))

    tracks = [
        node._track_hand(_make_image_msg(), _make_image_msg(), _make_info_msg(), FakeTransform())
        for _ in range(3)
    ]

    assert [t.fist for t in tracks] == [False, False, True]


def test_on_set_mode_track_hand_resets_fist_confirm_count(node):
    node._fist_confirm_count = 5

    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_HAND
    request.tool_class = ''
    node._on_set_mode(request, SetVisionMode.Response())

    assert node._fist_confirm_count == 0


def test_set_mode_track_tool_publishes_checkpoint(node):
    published = []
    node.pub_debug_events.publish = published.append
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_TOOL
    request.tool_class = 'spanner'

    node._on_set_mode(request, SetVisionMode.Response())

    assert len(published) == 1
    payload = json.loads(published[0].data)
    assert payload['phase'] == 'C'
    assert payload['checkpoint_id'] == 'vision_set_mode_track_tool'
    assert payload['status'] == 'PASS'


def test_set_mode_track_hand_publishes_checkpoint(node):
    published = []
    node.pub_debug_events.publish = published.append
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_HAND

    node._on_set_mode(request, SetVisionMode.Response())

    payload = json.loads(published[0].data)
    assert payload['phase'] == 'G'
    assert payload['checkpoint_id'] == 'vision_set_mode_track_hand'
    assert payload['status'] == 'PASS'


def test_set_mode_off_publishes_checkpoint(node):
    published = []
    node.pub_debug_events.publish = published.append
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.OFF

    node._on_set_mode(request, SetVisionMode.Response())

    payload = json.loads(published[0].data)
    assert payload['phase'] == 'K'
    assert payload['checkpoint_id'] == 'vision_set_mode_off'
    assert payload['status'] == 'PASS'


def test_track_tool_success_publishes_tool_track_valid_checkpoint(node):
    node.tf_buffer.lookup_transform = lambda *a, **k: None
    published = []
    node.pub_debug_events.publish = published.append
    # 이 테스트는 체크포인트 발행 로직만 검증 대상 - 디버그 이미지 인코딩(cv2.putText가
    # bgr8/uint8을 요구)까지 흉내내는 건 범위 밖이라 꺼서 우회한다
    # (다른 기존 테스트 test_track_tool_debug_image_off_when_disabled와 동일 패턴).
    node.publish_debug_image = False

    color_msg = _make_image_msg()
    depth_msg = _make_image_msg()
    info_msg = _make_info_msg()
    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 310, 230, 330, 250
    detection_msg = _make_detection_msg([detection])

    import numpy as np
    fake_depth = np.full((480, 424), 500, dtype=np.uint16)
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: fake_depth

    track = node._track_tool(
        color_msg, depth_msg, info_msg, detection_msg, FakeTransform(), 'spanner')

    assert track is not None
    checkpoint_payloads = [json.loads(p.data) for p in published]
    matches = [p for p in checkpoint_payloads if p['checkpoint_id'] == 'tool_track_valid']
    assert len(matches) == 1
    assert matches[0]['phase'] == 'C'
    assert matches[0]['status'] == 'PASS'
    assert matches[0]['data']['tool_class'] == 'spanner'
    # keypoint 없는 Detection2D는 bbox 모드 - mid가 아니라서 depth_valid는 False.
    assert matches[0]['data']['depth_valid'] is False


def test_track_tool_missing_target_does_not_publish_checkpoint(node):
    node.tf_buffer.lookup_transform = lambda *a, **k: None
    published = []
    node.pub_debug_events.publish = published.append
    node.tracker.update = lambda *a, **k: None
    # depth 이미지 읽기(_track_tool 최상단)가 tracker.update보다 먼저 실행되므로
    # 실제 cv_bridge가 빈 encoding으로 터지지 않게 최소 스텁이 필요하다. 디버그 이미지
    # 인코딩은 이 테스트 범위 밖이라 꺼서 우회한다(위 테스트와 동일 이유).
    node.publish_debug_image = False
    import numpy as np
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: np.zeros(
        (480, 424), dtype=np.uint16)

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(),
        _make_detection_msg([]), FakeTransform(), 'spanner')

    assert track is None
    assert published == []


def test_track_hand_success_publishes_hand_pose_checkpoint(node, monkeypatch):
    _setup_hand_depth(node, depth_mm=500)
    _patch_hand_landmarks(node, monkeypatch, _make_hand_landmarks(wrist_x=0.5, wrist_y=0.5, fist=False))
    published = []
    node.pub_debug_events.publish = published.append

    track = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), FakeTransform())

    assert track.detected is True
    payload = json.loads(published[-1].data)
    assert payload['phase'] == 'H'
    assert payload['checkpoint_id'] == 'hand_pose_published'
    assert payload['status'] == 'PASS'
