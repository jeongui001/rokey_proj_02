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


class _FakeResult:
    def __init__(self, success=True, message=''):
        self.success = success
        self.message = message
        self.measured_payload_kg = 0.0
        self.final_width_mm = 0.0
        self.grip_detected = False


class _FakeResponse:
    def __init__(self, result):
        self.result = result


class _FakeFuture:
    def __init__(self, response):
        self._response = response

    def result(self):
        return self._response


def test_user_command_ignored_unless_idle(node):
    node.state = State.SERVO_PICK
    called = []
    node._handle_parsing = lambda text: called.append(text)

    node._on_user_command(String(data='스패너 갖다줘'))

    assert called == []


def test_user_command_triggers_parsing_and_move_to_watch(node):
    node.state = State.IDLE
    node._call_llm = lambda text: {'tool': 'spanner', 'action': 'handover'}
    sent_goals = []
    node._send_robot_goal = lambda task_type, **kw: sent_goals.append((task_type, kw))
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_user_command(String(data='스패너 갖다줘'))

    assert node.state == State.MOVE_TO_WATCH
    assert node.current_tool == 'spanner'
    assert sent_goals == [('move_named', {'named_target': 'watch'})]


def test_parsing_failure_returns_to_idle(node):
    node.state = State.IDLE

    def _raise(text):
        raise NotImplementedError('todo')
    node._call_llm = _raise

    node._on_user_command(String(data='asdf'))

    assert node.state == State.IDLE


def test_move_to_watch_result_success_transitions_to_detect_track(node):
    node.state = State.MOVE_TO_WATCH

    node._on_robot_result(_FakeFuture(_FakeResponse(_FakeResult(success=True))))

    assert node.state == State.DETECT_TRACK


def test_move_to_watch_result_failure_transitions_to_fault(node):
    node.state = State.MOVE_TO_WATCH

    node._on_robot_result(_FakeFuture(_FakeResponse(_FakeResult(success=False, message='motion failed'))))

    assert node.state == State.FAULT
