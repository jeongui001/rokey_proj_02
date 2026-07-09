import time

import pytest
import rclpy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, String

from handover_interfaces.msg import GripperState
from operator_gui.ros_client import DEFAULT_CAMERA_TOPIC, RosClient, _OperatorGuiNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _FakeOwner:
    def __init__(self):
        self.task_status_calls = []
        self.gripper_state_calls = []
        self.fault_calls = []
        self.camera_image_calls = []
        self.mic_level_calls = []
        self.stt_status_calls = []
        self.stt_command_calls = []
        self.debug_event_calls = []
        self.on_task_status = lambda state, detail, mode, safety, resumable: (
            self.task_status_calls.append((state, detail, mode, safety, resumable)))
        self.on_gripper_state = lambda width, grip: self.gripper_state_calls.append((width, grip))
        self.on_fault = lambda msg: self.fault_calls.append(msg)
        self.on_camera_image = lambda data: self.camera_image_calls.append(data)
        self.on_mic_level = lambda level: self.mic_level_calls.append(level)
        self.on_stt_status = lambda state, detail, data: self.stt_status_calls.append(
            (state, detail, data))
        self.on_stt_command = lambda text: self.stt_command_calls.append(text)
        self.on_debug_event = lambda payload: self.debug_event_calls.append(payload)


@pytest.fixture
def owner():
    return _FakeOwner()


@pytest.fixture
def node(owner):
    n = _OperatorGuiNode(owner, DEFAULT_CAMERA_TOPIC)
    n.subscribe_all()
    yield n
    n.destroy_node()


@pytest.fixture
def peer():
    p = rclpy.create_node('test_peer_node')
    yield p
    p.destroy_node()


def _spin_until(spin_target, predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(spin_target, timeout_sec=0.1)
        if predicate():
            return True
    return False


def test_task_status_callback_parses_json(node, owner, peer):
    pub = peer.create_publisher(String, '/task/status', 10)
    time.sleep(0.3)
    pub.publish(String(
        data='{"state": "IDLE", "detail": "ready", '
             '"operation_mode": "AUTO", "safety_state": "NORMAL"}'))

    assert _spin_until(node, lambda: owner.task_status_calls)
    assert owner.task_status_calls == [('IDLE', 'ready', 'AUTO', 'NORMAL', False)]


def test_gripper_state_callback_forwards_fields(node, owner, peer):
    pub = peer.create_publisher(GripperState, '/gripper/state', 10)
    time.sleep(0.3)
    pub.publish(GripperState(width_mm=30.0, grip_detected=True))

    assert _spin_until(node, lambda: owner.gripper_state_calls)
    assert owner.gripper_state_calls == [(30.0, True)]


def test_fault_callback_forwards_message(node, owner, peer):
    pub = peer.create_publisher(String, '/robot/fault', 10)
    time.sleep(0.3)
    pub.publish(String(data='torque anomaly'))

    assert _spin_until(node, lambda: owner.fault_calls)
    assert owner.fault_calls == ['torque anomaly']


def test_camera_image_callback_forwards_raw_bytes(node, owner, peer):
    pub = peer.create_publisher(CompressedImage, DEFAULT_CAMERA_TOPIC, 10)
    time.sleep(0.3)
    msg = CompressedImage()
    msg.format = 'jpeg'
    msg.data = [1, 2, 3, 4]
    pub.publish(msg)

    assert _spin_until(node, lambda: owner.camera_image_calls)
    assert owner.camera_image_calls == [bytes([1, 2, 3, 4])]


def test_mic_level_callback_forwards_value(node, owner, peer):
    pub = peer.create_publisher(Float32, '/stt/mic_level', 10)
    time.sleep(0.3)
    pub.publish(Float32(data=0.42))

    assert _spin_until(node, lambda: owner.mic_level_calls)
    assert owner.mic_level_calls == pytest.approx([0.42])


def test_stt_command_callback_forwards_text(node, owner, peer):
    pub = peer.create_publisher(String, '/user_command/text', 10)
    time.sleep(0.3)
    pub.publish(String(data='스패너 갖다줘'))

    assert _spin_until(node, lambda: owner.stt_command_calls)
    assert owner.stt_command_calls == ['스패너 갖다줘']


def test_stt_status_callback_parses_json(node, owner, peer):
    pub = peer.create_publisher(String, '/stt/status', 10)
    time.sleep(0.3)
    pub.publish(String(data='{"state": "wakeword_detected", "detail": "말해주세요", "data": {"rms": 10}}'))

    assert _spin_until(node, lambda: owner.stt_status_calls)
    assert owner.stt_status_calls == [('wakeword_detected', '말해주세요', {'rms': 10})]


def test_debug_event_callback_parses_json(node, owner, peer):
    pub = peer.create_publisher(String, '/debug/events', 10)
    time.sleep(0.3)
    pub.publish(String(
        data='{"node": "vision_node", "level": "WARN", '
             '"category": "TRACK_TOOL", "reason": "target_missing"}'))

    assert _spin_until(node, lambda: owner.debug_event_calls)
    assert owner.debug_event_calls == [{
        'node': 'vision_node',
        'level': 'WARN',
        'category': 'TRACK_TOOL',
        'reason': 'target_missing',
    }]


def test_publish_command_sends_message(node, peer):
    received = []
    peer.create_subscription(String, '/user_command/text', lambda m: received.append(m.data), 10)
    time.sleep(0.3)

    assert node.publish_command('스패너 갖다줘') is True

    assert _spin_until(peer, lambda: received)
    assert received == ['스패너 갖다줘']


def test_publish_command_rejects_empty_text(node):
    assert node.publish_command('   ') is False


@pytest.fixture
def client():
    c = RosClient()
    yield c
    c.close()


def test_connect_starts_and_close_stops_spin_thread(client):
    client.connect()
    assert client.is_connected() is True

    client.close()
    assert client.is_connected() is False


def test_connect_notifies_connection_changed(client):
    received = []
    client.on_connection_changed = lambda connected: received.append(connected)

    client.connect()

    assert received == [True]


def test_ensure_connected_restarts_dead_spin_thread(client):
    received = []
    client.on_connection_changed = lambda connected: received.append(connected)
    client.connect()
    client._spin_thread.stop()

    client.ensure_connected()

    assert client.is_connected() is True
    assert received == [True, False, True]


def test_subscribe_all_and_receive_task_status(client, peer):
    received = []
    client.on_task_status = lambda state, detail, mode, safety, resumable: received.append(
        (state, detail, mode, safety, resumable))
    client.subscribe_all()
    client.connect()

    pub = peer.create_publisher(String, '/task/status', 10)
    time.sleep(0.3)
    pub.publish(String(
        data='{"state": "IDLE", "detail": "ready", '
             '"operation_mode": "AUTO", "safety_state": "NORMAL"}'))

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not received:
        time.sleep(0.05)

    assert received == [('IDLE', 'ready', 'AUTO', 'NORMAL', False)]


def test_publish_command_via_client(client, peer):
    received = []
    peer.create_subscription(String, '/user_command/text', lambda m: received.append(m.data), 10)
    client.subscribe_all()
    client.connect()
    time.sleep(0.3)

    assert client.publish_command('스패너 갖다줘') is True

    assert _spin_until(peer, lambda: received)
    assert received == ['스패너 갖다줘']
