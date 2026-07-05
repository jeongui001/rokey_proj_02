import rclpy
import pytest

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from handover_interfaces.msg import ToolTrack, DetectionArray, Detection2D
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

    from geometry_msgs.msg import PoseStamped
    expected_pose = PoseStamped()
    node._track_hand = lambda color, depth, info, tf: expected_pose

    published = []
    node.pub_hand_pose.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert published == [expected_pose]


def test_track_tool_filters_by_class_and_reconstructs_position(node):
    node.tf_buffer.lookup_transform = lambda *a, **k: None

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
