import json

import rclpy
import pytest
from std_msgs.msg import String

from task_manager.task_manager_node import TaskManagerNode, State


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = TaskManagerNode()
    yield n
    n.destroy_node()


def test_initial_state_is_idle(node):
    assert node.state == State.IDLE


def test_set_state_publishes_json_status(node):
    published = []
    node.pub_status.publish = published.append

    node._set_state(State.PARSING, detail='hello')

    assert node.state == State.PARSING
    assert len(published) == 1
    payload = json.loads(published[0].data)
    assert payload == {'state': 'PARSING', 'detail': 'hello'}


def test_fault_message_transitions_to_fault_from_any_state(node):
    published = []
    node.pub_status.publish = published.append
    node.state = State.SERVO_PICK

    msg = String()
    msg.data = 'torque anomaly'
    node._on_fault(msg)

    assert node.state == State.FAULT
    payload = json.loads(published[-1].data)
    assert payload['detail'] == 'torque anomaly'


def test_fault_message_ignored_if_already_in_fault(node):
    node.state = State.FAULT
    published = []
    node.pub_status.publish = published.append

    msg = String()
    msg.data = 'another fault'
    node._on_fault(msg)

    assert published == []
