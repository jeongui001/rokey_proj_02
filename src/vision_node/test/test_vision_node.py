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
    import numpy as np

    n = VisionNode()
    # _on_synced_images가 TRACK_TOOL/TRACK_HAND에서 이제 _align_depth_msg를 거치는데,
    # 그게 뎁스->컬러 TF/intrinsics 캐시가 있어야 동작한다(실제로는 TF 조회로 채워짐).
    # 여기서 항등 변환(rotation=identity, translation=0) + 컬러와 동일한 intrinsics로
    # 미리 채워 두면 align_depth_to_color가 순수 통과(입력 그대로 반환)가 되어, 이 필드가
    # 생기기 전 테스트들이 가정하던 "depth_m_img == imgmsg_to_cv2 결과"와 동일하게 유지된다.
    n._depth_intrinsics = (600.0, 600.0, 320.0, 240.0)  # _make_info_msg()의 컬러 intrinsics와 동일
    n._depth_to_color = (np.eye(3), np.zeros(3))
    # _on_synced_images 디스패치만 검증하는 테스트들은 실제 이미지 데이터가 없어(_make_image_msg는
    # header만 채운 빈 Image) 진짜 _align_depth_msg가 cv_bridge 인코딩 에러로 실패한다 - 그
    # 테스트들의 관심사는 정렬 자체가 아니라 모드별 디스패치이므로 기본은 항등 통과로 스텁하고,
    # 정렬 로직 자체를 검증하는 테스트는 이걸 다시 원래 구현으로 되돌려 쓴다.
    n._align_depth_msg = lambda depth_msg, color_msg, info_msg: depth_msg
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


def test_set_mode_track_hand_enables_docker_hand_detection(node):
    published = []
    node.pub_hand_enable.publish = published.append
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_HAND

    node._on_set_mode(request, SetVisionMode.Response())

    assert [msg.data for msg in published] == [True]


def test_set_mode_track_tool_and_off_disable_docker_hand_detection(node):
    published = []
    node.pub_hand_enable.publish = published.append

    for mode in (SetVisionMode.Request.TRACK_TOOL, SetVisionMode.Request.OFF):
        request = SetVisionMode.Request()
        request.mode = mode
        node._on_set_mode(request, SetVisionMode.Response())

    assert [msg.data for msg in published] == [False, False]


def test_set_mode_track_hand_clears_stale_hand_detection(node):
    node._hand_detection = ({'detected': True}, node.get_clock().now())
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_HAND

    node._on_set_mode(request, SetVisionMode.Response())

    assert node._hand_detection is None


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


def test_synced_images_track_tool_align_failure_still_publishes_debug_image(node):
    """_align_depth_msg 실패(TF/camera_info 미비, cv2.rgbd 예외 등)는 3D 추적만 못 하는
    부분 열화여야 한다 - 524955d 회귀 전에는 여기서 곧장 return해 _track_tool 내부에서만
    이뤄지는 디버그 영상 발행까지 함께 끊겨 GUI가 "카메라 꺼짐"으로 보였다(팀원 실기 확인:
    태스크 시작 순간 카메라 화면 정지)."""
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._align_depth_msg = lambda *a, **k: None

    track_tool_calls = []
    node._track_tool = lambda *a, **k: track_tool_calls.append(1)

    debug_calls = []
    node._publish_debug_image = lambda color_msg, detections, chosen_det, axis_debug: debug_calls.append(
        (detections, chosen_det, axis_debug))

    detection = Detection2D()
    detection_msg = _make_detection_msg([detection])
    node._on_synced_images(_make_image_msg(), _make_image_msg(), _make_info_msg(), detection_msg)

    assert track_tool_calls == []  # depth 정렬 없이 3D 추적은 시도하지 않음
    assert debug_calls == [([detection], None, None)]  # 화면은 계속 흘려보냄


def test_synced_images_track_hand_align_failure_still_calls_track_hand(node):
    """TRACK_HAND도 같은 이유로 _align_depth_msg 실패 시 프레임을 드롭하지 않고, depth_msg=None
    으로 _track_hand를 그대로 호출해 매 프레임 HandTrack을 발행해야 한다 - 안 그러면
    /vision/hand_track이 조용히 멈춰 robot_control의 HandServoLoop가 "손 유실"과
    "vision_node 응답 없음"을 구분할 수 없다."""
    node.mode = SetVisionMode.Request.TRACK_HAND
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._align_depth_msg = lambda *a, **k: None

    calls = []

    def _track_hand(color, depth, info, tf):
        calls.append(depth)
        track = HandTrack()
        track.detected = False
        return track

    node._track_hand = _track_hand
    published = []
    node.pub_hand_track.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert calls == [None]  # depth 정렬 실패가 그대로 전달됨
    assert len(published) == 1


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


def test_synced_images_off_mode_publishes_debug_image_when_tf_lookup_fails(node):
    """camera->base TF는 _track_tool/_track_hand의 3D 변환에만 필요하다 - 조회 실패로
    프레임 전체를 드롭하면 OFF 모드조차 디버그 영상을 못 내보낸다(524955d 이후 회귀,
    2026-07-13 실기: 태스크 시작 직후 로봇이 처음 움직이는 전이 구간에서 재현). 모드
    분기 자체에는 도달해야 한다."""
    from tf2_ros import TransformException

    def _raise(*a, **k):
        raise TransformException('no tf yet')

    node.mode = SetVisionMode.Request.OFF
    node.tf_buffer.lookup_transform = _raise

    calls = []
    node._publish_debug_image = lambda color_msg, detections, chosen_det, axis_debug: calls.append(
        (detections, chosen_det, axis_debug))

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert calls == [([], None, None)]


def test_synced_images_track_tool_publishes_debug_image_when_tf_lookup_fails(node):
    """TRACK_TOOL에서도 TF 실패는 3D 추적만 못 하는 부분 열화여야 한다 - _align_depth_msg
    호출조차 시도하지 않고(tf_at_stamp 없이는 결과를 못 쓰므로) 곧장 디버그 영상 폴백으로
    간다."""
    from tf2_ros import TransformException

    def _raise(*a, **k):
        raise TransformException('no tf yet')

    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = _raise

    align_calls = []
    node._align_depth_msg = lambda *a, **k: align_calls.append(1)

    debug_calls = []
    node._publish_debug_image = lambda color_msg, detections, chosen_det, axis_debug: debug_calls.append(
        (detections, chosen_det, axis_debug))

    detection = Detection2D()
    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg([detection]))

    assert align_calls == []  # tf_at_stamp 없이는 정렬 결과를 쓸 수 없으므로 시도조차 안 함
    assert debug_calls == [([detection], None, None)]


def test_synced_images_track_hand_dispatches_when_tf_lookup_fails(node):
    """TRACK_HAND도 TF 실패 시 모드 분기 자체에는 도달해야 한다 - _track_hand는
    depth_msg=None으로 호출돼 detected=False로 발행하고, 디버그 영상도 계속 나간다."""
    from tf2_ros import TransformException

    def _raise(*a, **k):
        raise TransformException('no tf yet')

    node.mode = SetVisionMode.Request.TRACK_HAND
    node.tf_buffer.lookup_transform = _raise

    calls = []

    def _track_hand(color, depth, info, tf):
        calls.append((depth, tf))
        track = HandTrack()
        track.detected = False
        return track

    node._track_hand = _track_hand
    published = []
    node.pub_hand_track.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert calls == [(None, None)]
    assert len(published) == 1


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


def test_synced_images_track_hand_also_publishes_raw_debug_image(node):
    # TRACK_HAND(핸드오버 접근/대기) 구간에서도 OFF와 동일한 이유로 디버그 영상은
    # 계속 흘려보내야 한다 - 안 그러면 이 구간 내내 GUI가 "카메라 영상이 멈췄습니다"로
    # 표시해 실제 고장과 구분이 안 된다(2026-07-12 확인).
    node.mode = SetVisionMode.Request.TRACK_HAND
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._track_hand = lambda color, depth, info, tf: None

    calls = []
    node._publish_debug_image = lambda color_msg, detections, chosen_det, axis_debug: calls.append(
        (detections, chosen_det, axis_debug))

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert calls == [([], None, None)]


def test_synced_images_track_hand_skips_debug_image_when_disabled(node):
    node.mode = SetVisionMode.Request.TRACK_HAND
    node.publish_debug_image = False
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._track_hand = lambda color, depth, info, tf: None

    calls = []
    node._publish_debug_image = lambda *a, **k: calls.append(1)

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert calls == []


def test_on_depth_info_caches_intrinsics(node):
    node._depth_intrinsics = None
    info = _make_info_msg()  # k = [600,0,320, 0,600,240, 0,0,1]

    node._on_depth_info(info)

    assert node._depth_intrinsics == (600.0, 600.0, 320.0, 240.0)


def test_get_depth_to_color_extrinsics_caches_after_first_lookup(node):
    """뎁스->컬러 외부파라미터는 카메라 리그에 고정된 물리적 값이라 최초 1회만 TF를
    조회하고 이후로는 캐시를 그대로 써야 한다(매 프레임 TF 버퍼 조회 비용을 아낌)."""
    import numpy as np

    node._depth_to_color = None
    calls = []

    def _lookup(target, source, time, timeout=None):
        calls.append((target, source))
        tf = FakeTransform()
        tf.transform.translation.x = 0.015
        return tf

    node.tf_buffer.lookup_transform = _lookup
    depth_msg = _make_image_msg()
    depth_msg.header.frame_id = 'camera_depth_optical_frame'
    color_msg = _make_image_msg()
    color_msg.header.frame_id = 'camera_color_optical_frame'

    first = node._get_depth_to_color_extrinsics(depth_msg, color_msg)
    second = node._get_depth_to_color_extrinsics(depth_msg, color_msg)

    assert len(calls) == 1  # 두 번째 호출은 TF를 다시 조회하지 않음
    assert calls[0] == ('camera_color_optical_frame', 'camera_depth_optical_frame')
    rotation, translation = first
    assert np.array_equal(rotation, np.eye(3))
    assert translation[0] == pytest.approx(0.015)
    assert second is first  # 캐시된 동일 객체


def test_get_depth_to_color_extrinsics_returns_none_when_tf_missing(node):
    from tf2_ros import TransformException

    node._depth_to_color = None
    node.tf_buffer.lookup_transform = lambda *a, **k: (_ for _ in ()).throw(
        TransformException('no tf yet'))

    result = node._get_depth_to_color_extrinsics(_make_image_msg(), _make_image_msg())

    assert result is None
    assert node._depth_to_color is None


def test_align_depth_msg_produces_color_grid_aligned_depth(node):
    """실제 _align_depth_msg(스텁 없이) 구현 검증 - depth intrinsics/TF 캐시를 그대로
    쓰고 align_depth_to_color 결과를 컬러와 같은 해상도의 Image msg로 되돌려주는지."""
    import numpy as np
    del node._align_depth_msg  # fixture의 기본 스텁을 걷어내고 실제 구현을 쓴다

    node._depth_intrinsics = (600.0, 600.0, 50.0, 50.0)
    node._depth_to_color = (np.eye(3), np.zeros(3))

    raw_depth_mm = np.zeros((100, 100), dtype=np.uint16)
    raw_depth_mm[50, 50] = 500  # 0.5m, depth_ppx/ppy(50,50)와 일치 -> 자기 자신에게 재투영
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: raw_depth_mm

    depth_msg = _make_image_msg()
    color_msg = _make_image_msg()
    color_msg.height, color_msg.width = 100, 100
    info_msg = _make_info_msg()
    info_msg.k = [600.0, 0.0, 50.0, 0.0, 600.0, 50.0, 0.0, 0.0, 1.0]

    aligned_msg = node._align_depth_msg(depth_msg, color_msg, info_msg)

    assert aligned_msg is not None
    assert aligned_msg.header == color_msg.header
    decoded = node._bridge.imgmsg_to_cv2(aligned_msg, desired_encoding='passthrough')
    assert decoded[50, 50] == 500


def test_align_depth_msg_returns_none_when_extrinsics_missing(node):
    from tf2_ros import TransformException

    del node._align_depth_msg
    node._depth_to_color = None
    node.tf_buffer.lookup_transform = lambda *a, **k: (_ for _ in ()).throw(
        TransformException('no tf yet'))

    result = node._align_depth_msg(_make_image_msg(), _make_image_msg(), _make_info_msg())

    assert result is None


def test_safe_call_catches_generic_exception_and_logs(node):
    """예전엔 NotImplementedError만 잡아서 실제 예외(ValueError 등)가 그대로 새어나가
    vision_node 프로세스 전체를 죽였다(기본 SingleThreadedExecutor가 콜백 예외를 스핀
    루프에서 재발생시킴, respawn도 없어 카메라 송출이 영구히 멈추는 원인이었다).
    지금은 모든 Exception을 잡아 이번 프레임만 건너뛰어야 한다."""
    def _boom():
        raise ValueError('boom')

    logged = []
    node.get_logger().error = lambda msg, **k: logged.append(msg)

    result = node._safe_call(_boom, default='fallback')

    assert result == 'fallback'
    assert len(logged) == 1
    assert 'boom' in logged[0]


def test_safe_call_still_returns_default_on_not_implemented_error(node):
    def _boom():
        raise NotImplementedError('not yet')

    node.get_logger().error = lambda msg, **k: None
    result = node._safe_call(_boom, default='fallback')

    assert result == 'fallback'


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


def test_track_hand_none_depth_msg_keeps_fist_but_not_detected(node, monkeypatch):
    """_align_depth_msg가 실패해 depth_msg=None으로 넘어와도(524955d 회귀 - TF/camera_info
    미비, cv2.rgbd 예외 등) 3D 위치만 못 쓸 뿐 2D 검출/주먹 판정은 이미 끝난 상태이므로
    그대로 반영해 발행해야 한다(_track_hand는 절대 None을 반환하지 않는 설계)."""
    detection = ((212, 120), [(0.5, 0.5)] * 21, 0.9)
    _prime_track_hand(node, monkeypatch, detection=detection, fist=True)
    node.fist_confirm_frames = 1
    node._fist_counter = 0

    track = node._track_hand(
        _make_image_msg(), None, _make_info_msg(), 'fake_tf')

    assert track.detected is False
    assert track.fist is True
    assert track.confidence == pytest.approx(0.9)


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
    assert track.yaw_valid is True


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
    # identity로 남은 이유가 "yaw=0으로 측정됨"이 아니라 "측정 실패"임을 구독측이
    # 구분할 수 있어야 한다 - robot_control.ServoLoop이 이 플래그로 hold 처리한다.
    assert track.yaw_valid is False


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
