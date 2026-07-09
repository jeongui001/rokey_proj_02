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
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: fake_depth

    track = node._track_tool(
        color_msg, depth_msg, info_msg, detection_msg, FakeTransform(), 'spanner')

    assert track is not None
    assert track.tool_class == 'spanner'
    assert track.pose.position.z == pytest.approx(0.5, abs=1e-3)
    assert track.depth_valid is True
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
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: fake_depth

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


def test_track_tool_orientation_identity_when_axis_unavailable(node):
    """뎁스가 전부 무효(0)라 축을 못 구하면 orientation은 identity로 남아야 한다."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.zeros((480, 424), dtype=np.uint16)  # 전부 뎁스 구멍
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: fake_depth

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 100, 100, 200, 200
    detection_msg = _make_detection_msg([detection])

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is not None
    assert track.depth_valid is False
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
