import json

import rclpy
import pytest

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from handover_interfaces.msg import ToolTrack, DetectionArray, Detection2D, HandTrack
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


def test_synced_images_off_mode_still_publishes_raw_debug_image(node):
    # mode==OFF에서도 GUI가 카메라 연결을 상시 확인할 수 있도록 인식 박스 없는
    # 원본 프리뷰는 계속 흘려보낸다 (ToolTrack/HandTrack은 발행하지 않음).
    node.mode = SetVisionMode.Request.OFF
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'

    calls = []
    node._publish_debug_image = lambda color_msg, detections, chosen_det, axis_debug: calls.append(
        (detections, chosen_det, axis_debug))

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert calls == [([], None, None)]


def test_synced_images_off_mode_skips_debug_image_when_disabled(node):
    node.mode = SetVisionMode.Request.OFF
    node.publish_debug_image = False
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'

    calls = []
    node._publish_debug_image = lambda *a, **k: calls.append(1)

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert calls == []


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


def _prime_track_hand(node, monkeypatch, detection, fist, depth=(0.5, 1.0)):
    """_track_hand의 depth 변환/tf 의존을 모의로 대체하고, 컨테이너(hand_track_docker_node)가
    보낸 것처럼 _hand_detection 캐시를 직접 채운다."""
    import numpy as np
    import vision_node.vision_node as vn

    monkeypatch.setattr(
        node._bridge, 'imgmsg_to_cv2',
        lambda msg, desired_encoding=None: np.zeros((240, 424, 3), dtype=np.uint8))
    monkeypatch.setattr(vn, 'patch_median_depth', lambda *a, **k: depth)
    node._tf_matrix = lambda tf: [[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0]]
    if detection is None:
        payload = {'detected': False}
    else:
        (px, py), _landmarks, confidence = detection
        payload = {'detected': True, 'palm_px': [px, py], 'confidence': confidence, 'is_fist': fist}
    node._hand_detection = (payload, node.get_clock().now())


def test_track_hand_none_when_no_hand(node, monkeypatch):
    _prime_track_hand(node, monkeypatch, detection=None, fist=False)
    node._fist_counter = 5  # 손이 사라지면 리셋되는지 확인

    track = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), 'fake_tf')

    assert track.detected is False
    assert track.fist is False
    assert track.header.frame_id == 'base_link'
    assert node._fist_counter == 0


def test_track_hand_detected_with_position_and_fist_debounce(node, monkeypatch):
    detection = ((212, 120), [(0.5, 0.5)] * 21, 0.9)
    _prime_track_hand(node, monkeypatch, detection=detection, fist=True)
    node.fist_confirm_frames = 2
    node._fist_counter = 0

    t1 = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), 'fake_tf')
    assert t1.detected is True
    assert t1.header.frame_id == 'base_link'
    assert t1.confidence == pytest.approx(0.9)
    assert t1.pose.position.z == pytest.approx(0.5)  # patched depth
    assert t1.fist is False  # 1프레임: 아직 확정 전(confirm_frames=2)

    t2 = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), 'fake_tf')
    assert t2.fist is True  # 2프레임 연속 -> 확정


def test_track_hand_invalid_depth_keeps_fist_but_not_detected(node, monkeypatch):
    detection = ((212, 120), [(0.5, 0.5)] * 21, 0.9)
    # depth 무효(z_m=None) - 위치는 못 쓰지만 주먹 신호는 유지되어야 한다
    _prime_track_hand(node, monkeypatch, detection=detection, fist=True, depth=(None, 0.0))
    node.fist_confirm_frames = 1
    node._fist_counter = 0

    track = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), 'fake_tf')

    assert track.detected is False
    assert track.fist is True


def test_on_hand_track_docker_caches_payload(node):
    import json as _json
    from std_msgs.msg import String

    msg = String()
    msg.data = _json.dumps({'detected': True, 'palm_px': [1, 2], 'confidence': 0.5, 'is_fist': False})
    node._on_hand_track_docker(msg)

    assert node._hand_detection is not None
    payload, _received_at = node._hand_detection
    assert payload['palm_px'] == [1, 2]
    assert payload['is_fist'] is False


def test_on_hand_track_docker_ignores_invalid_json(node):
    from std_msgs.msg import String

    node._hand_detection = None
    msg = String()
    msg.data = 'not json'
    node._on_hand_track_docker(msg)

    assert node._hand_detection is None


def test_track_hand_stale_detection_is_not_detected(node, monkeypatch):
    import time

    detection = ((212, 120), [(0.5, 0.5)] * 21, 0.9)
    _prime_track_hand(node, monkeypatch, detection=detection, fist=True)
    node.hand_detection_max_age_s = 0.0  # 캐시가 있어도 즉시 stale 처리되게
    node._fist_counter = 3
    time.sleep(0.01)

    track = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), 'fake_tf')

    assert track.detected is False
    assert node._fist_counter == 0


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
    """뎁스가 전부 무효(0)라 축을 못 구하면 orientation은 identity로 남아야 한다.

    2026-07-08 실기 사고 이후: 추적 사이클 첫 프레임부터 depth가 무효면 last_valid_z도
    없어 이제는 후보 자체가 버려진다(track is None, 아래
    test_track_tool_returns_none_on_first_frame_with_fully_invalid_depth 참고) - 그래서
    이 테스트는 "마지막 유효 z 유지" 동작을 보려면 먼저 유효 프레임으로 한 번
    프라이밍한다."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    valid_depth = np.full((480, 424), 500, dtype=np.uint16)  # 0.5m, 유효 프라이밍용
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: valid_depth

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 100, 100, 200, 200
    detection_msg = _make_detection_msg([detection])

    primed = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')
    assert primed is not None
    # keypoint 없는 Detection2D는 bbox 모드 - mid가 아니라서 published depth_valid는
    # False로 강제되지만(2026-07-11), 패치 자체는 유효했으므로 last_valid_z는 내부적으로
    # 채워진다(아래에서 depth 전부 무효인 프레임에 이 값으로 고정되는지로 확인).
    assert primed.depth_valid is False

    fake_depth = np.zeros((480, 424), dtype=np.uint16)  # 전부 뎁스 구멍
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: fake_depth

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is not None
    assert track.depth_valid is False
    assert track.pose.orientation.w == pytest.approx(1.0)
    assert track.pose.orientation.z == pytest.approx(0.0)


def test_track_tool_returns_none_on_first_frame_with_fully_invalid_depth(node):
    """2026-07-08 실기 사고 회귀 테스트: 추적 사이클의 첫 프레임부터 depth가 전부
    무효면(last_valid_z도 아직 없음) 좌표를 지어내지 말고 ToolTrack을 아예 발행하지
    않아야 한다 - 예전에는 z=0.0(카메라 장착 위치 근방)으로 폴백해 엉뚱한 좌표가
    나갔었다(망치가 보이는데도 로봇이 바닥을 내려찍은 원인)."""
    import numpy as np
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    fake_depth = np.zeros((480, 424), dtype=np.uint16)  # 전부 뎁스 구멍, 프라이밍 없음
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: fake_depth

    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 100, 100, 200, 200
    detection_msg = _make_detection_msg([detection])

    track = node._track_tool(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg,
        FakeTransform(), 'spanner')

    assert track is None


def test_publish_debug_image_overlays_position_when_given(node):
    """2026-07-08 실기 사고 이후 추가: position/depth_valid가 주어지면 계산된 base_link
    좌표를 화면에 같이 찍어야 한다 - rqt_image_view만 보고도 좌표 계산이 이상값인지
    바로 판단할 수 있게 하기 위함."""
    import numpy as np
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: np.zeros(
        (480, 424, 3), dtype=np.uint8)

    published = []
    node.pub_debug_image.publish = published.append

    node._publish_debug_image(
        _make_image_msg(), [], None, None, position=(0.1, 0.2, 0.3), depth_valid=True)

    assert len(published) == 1
    assert published[0].format == 'jpeg'
    assert len(published[0].data) > 0


def test_publish_debug_image_no_position_overlay_when_position_none(node):
    """position이 없으면(타겟 클래스 미검출 프레임) 오버레이 없이도 정상 발행돼야 한다."""
    import numpy as np
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: np.zeros(
        (480, 424, 3), dtype=np.uint8)

    published = []
    node.pub_debug_image.publish = published.append

    node._publish_debug_image(_make_image_msg(), [], None, None)

    assert len(published) == 1


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
# ---- _on_set_mode / _track_tool / _track_hand 체크포인트 발행 ----

def test_on_set_mode_track_hand_resets_fist_confirm_count(node):
    node._fist_counter = 5

    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_HAND
    request.tool_class = ''
    node._on_set_mode(request, SetVisionMode.Response())

    assert node._fist_counter == 0


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
    detection = ((212, 120), [(0.5, 0.5)] * 21, 0.9)
    _prime_track_hand(node, monkeypatch, detection=detection, fist=False)
    published = []
    node.pub_debug_events.publish = published.append

    track = node._track_hand(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), 'fake_tf')

    assert track.detected is True
    payload = json.loads(published[-1].data)
    assert payload['phase'] == 'H'
    assert payload['checkpoint_id'] == 'hand_pose_published'
    assert payload['status'] == 'PASS'
