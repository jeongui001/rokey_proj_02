import time

import pytest
import rclpy
from std_msgs.msg import String

from handover_interfaces.msg import GripperState
from handover_ui.ros_client import RosClient, _HandoverUiNode


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
        self.on_task_status = lambda state, detail: self.task_status_calls.append((state, detail))
        self.on_gripper_state = lambda width, grip: self.gripper_state_calls.append((width, grip))
        self.on_fault = lambda msg: self.fault_calls.append(msg)


@pytest.fixture
def owner():
    return _FakeOwner()


@pytest.fixture
def node(owner):
    n = _HandoverUiNode(owner)
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
    pub.publish(String(data='{"state": "IDLE", "detail": "ready"}'))

    assert _spin_until(node, lambda: owner.task_status_calls)
    assert owner.task_status_calls == [('IDLE', 'ready')]


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


def test_publish_command_sends_message(node, peer):
    received = []
    peer.create_subscription(String, '/user_command/text', lambda m: received.append(m.data), 10)
    time.sleep(0.3)

    node.publish_command('스패너 갖다줘')

    assert _spin_until(peer, lambda: received)
    assert received == ['스패너 갖다줘']


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


def test_subscribe_all_and_receive_task_status(client, peer):
    received = []
    client.on_task_status = lambda state, detail: received.append((state, detail))
    client.subscribe_all()
    client.connect()

    pub = peer.create_publisher(String, '/task/status', 10)
    time.sleep(0.3)
    pub.publish(String(data='{"state": "IDLE", "detail": "ready"}'))

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not received:
        time.sleep(0.05)

    assert received == [('IDLE', 'ready')]


def test_publish_command_via_client(client, peer):
    received = []
    peer.create_subscription(String, '/user_command/text', lambda m: received.append(m.data), 10)
    client.subscribe_all()
    client.connect()
    time.sleep(0.3)

    client.publish_command('스패너 갖다줘')

    assert _spin_until(peer, lambda: received)
    assert received == ['스패너 갖다줘']
