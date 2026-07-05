import json

import rclpy
import pytest
from std_msgs.msg import String

from geometry_msgs.msg import PoseStamped
from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode
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


def test_tool_track_ignored_unless_detect_track(node):
    node.state = State.IDLE
    node._check_trigger = lambda msg: True
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_tool_track(ToolTrack())

    assert sent == []


def test_tool_track_trigger_sends_servo_pick_goal(node):
    node.state = State.DETECT_TRACK
    node.current_tool = 'spanner'
    node._check_trigger = lambda msg: True
    node._get_grasp_spec = lambda tool_class: (30.0, 20.0)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_tool_track(ToolTrack())

    assert node.state == State.SERVO_PICK
    assert sent == [('servo_pick', {
        'tool_class': 'spanner', 'grasp_width_mm': 30.0, 'grasp_force_n': 20.0})]


def test_tool_track_no_trigger_increments_cycle_and_reports_after_max(node):
    node.state = State.DETECT_TRACK
    node._check_trigger = lambda msg: False

    node._on_tool_track(ToolTrack())
    assert node.state == State.DETECT_TRACK
    node._on_tool_track(ToolTrack())
    assert node.state == State.DETECT_TRACK
    node._on_tool_track(ToolTrack())

    assert node.state == State.IDLE


def test_servo_pick_result_torque_anomaly_goes_to_fault(node):
    node.state = State.SERVO_PICK

    node._handle_servo_pick_result(_FakeResult(success=False, message='torque anomaly'))

    assert node.state == State.FAULT


def test_servo_pick_result_other_failure_returns_to_detect_track(node):
    node.state = State.SERVO_PICK
    node._detect_track_cycles = 2

    node._handle_servo_pick_result(_FakeResult(success=False, message='timeout'))

    assert node.state == State.DETECT_TRACK
    assert node._detect_track_cycles == 0


def test_servo_pick_result_success_and_verify_passes_moves_to_move_safe(node):
    node.state = State.SERVO_PICK
    node._verify_grasp = lambda result: True
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_servo_pick_result(_FakeResult(success=True))

    assert node.state == State.MOVE_SAFE
    assert sent == [('move_named', {'named_target': 'safe'})]


def test_servo_pick_result_success_and_verify_fails_sends_release_and_retry(node):
    node.state = State.SERVO_PICK
    node._verify_grasp = lambda result: False
    node._verify_grasp_retries = 0
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_servo_pick_result(_FakeResult(success=True))

    assert node.state == State.VERIFY_GRASP
    assert sent == [('release_and_retry', {})]
    assert node._verify_grasp_retries == 1


def test_verify_grasp_exceeds_max_retries_reports_to_idle(node):
    node.state = State.SERVO_PICK
    node._verify_grasp = lambda result: False
    node._verify_grasp_retries = 2
    node._send_robot_goal = lambda *a, **k: None

    node._handle_servo_pick_result(_FakeResult(success=True))

    assert node.state == State.IDLE


def test_release_and_retry_result_success_returns_to_detect_track(node):
    node.state = State.VERIFY_GRASP

    node._handle_release_and_retry_result(_FakeResult(success=True))

    assert node.state == State.DETECT_TRACK


def test_release_and_retry_result_failure_goes_to_fault(node):
    node.state = State.VERIFY_GRASP

    node._handle_release_and_retry_result(_FakeResult(success=False, message='release failed'))

    assert node.state == State.FAULT


def test_move_safe_result_success_transitions_to_track_hand(node):
    node.state = State.MOVE_SAFE
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_move_safe_result(_FakeResult(success=True))

    assert node.state == State.TRACK_HAND
    assert node._hand_timeout_timer is not None
    node._hand_timeout_timer.cancel()


def test_move_safe_result_failure_goes_to_fault(node):
    node.state = State.MOVE_SAFE

    node._handle_move_safe_result(_FakeResult(success=False, message='motion failed'))

    assert node.state == State.FAULT


def test_hand_pose_sends_move_pose_with_offset(node):
    node.state = State.TRACK_HAND
    node._hand_timeout_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    msg = PoseStamped()
    msg.pose.position.z = 0.30
    node._on_hand_pose(msg)

    assert node._hand_timeout_timer is None
    assert sent[0][0] == 'move_pose'
    assert abs(sent[0][1]['target_pose'].pose.position.z - 0.38) < 1e-9


def test_hand_pose_ignored_unless_track_hand(node):
    node.state = State.IDLE
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_hand_pose(PoseStamped())

    assert sent == []


def test_hand_timeout_sends_fallback_goal(node):
    node.state = State.TRACK_HAND
    node._hand_timeout_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_hand_timeout()

    assert sent == [('move_named', {'named_target': 'handover_default'})]


def test_track_hand_result_success_transitions_to_wait_pull(node):
    node.state = State.TRACK_HAND
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_track_hand_result(_FakeResult(success=True))

    assert node.state == State.WAIT_PULL
    assert sent == [('handover_hold', {})]
    node._wait_pull_timeout_timer.cancel()


def test_wait_pull_result_success_goes_home(node):
    node.state = State.WAIT_PULL
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)
    node._set_vision_mode = lambda mode, tool_class='': None
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_wait_pull_result(_FakeResult(success=True, message='pull_detected, released'))

    assert node.state == State.HOME
    assert sent == [('move_named', {'named_target': 'home'})]


def test_wait_pull_timeout_sends_place_down(node):
    node.state = State.WAIT_PULL
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_wait_pull_timeout()

    assert node.state == State.RELEASE
    assert sent == [('place_down', {'named_target': 'place_down'})]


def test_release_result_success_goes_home(node):
    node.state = State.RELEASE
    node._set_vision_mode = lambda mode, tool_class='': None
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_release_result(_FakeResult(success=True))

    assert node.state == State.HOME
    assert sent == [('move_named', {'named_target': 'home'})]


def test_home_result_success_returns_to_idle(node):
    node.state = State.HOME
    node.current_tool = 'spanner'

    node._handle_home_result(_FakeResult(success=True))

    assert node.state == State.IDLE


def test_home_result_failure_goes_to_fault(node):
    node.state = State.HOME

    node._handle_home_result(_FakeResult(success=False, message='motion failed'))

    assert node.state == State.FAULT


class _FakeSetModeResponse:
    def __init__(self, success=True, message=''):
        self.success = success
        self.message = message


class _FakeSetModeFuture:
    def __init__(self, response):
        self._response = response

    def result(self):
        return self._response

    def add_done_callback(self, callback):
        callback(self)


def test_set_vision_mode_success_does_not_change_state(node):
    node.state = State.MOVE_TO_WATCH
    node.set_mode_client.call_async = lambda req: _FakeSetModeFuture(
        _FakeSetModeResponse(success=True))

    node._set_vision_mode(SetVisionMode.Request.TRACK_TOOL, 'spanner')

    assert node.state == State.MOVE_TO_WATCH


def test_set_vision_mode_failure_for_track_tool_goes_to_fault(node):
    node.state = State.MOVE_TO_WATCH
    node.set_mode_client.call_async = lambda req: _FakeSetModeFuture(
        _FakeSetModeResponse(success=False, message='camera busy'))

    node._set_vision_mode(SetVisionMode.Request.TRACK_TOOL, 'spanner')

    assert node.state == State.FAULT


def test_set_vision_mode_failure_for_track_hand_goes_to_fault(node):
    node.state = State.MOVE_SAFE
    node.set_mode_client.call_async = lambda req: _FakeSetModeFuture(
        _FakeSetModeResponse(success=False, message='camera busy'))

    node._set_vision_mode(SetVisionMode.Request.TRACK_HAND)

    assert node.state == State.FAULT


def test_set_vision_mode_failure_for_off_does_not_change_state(node):
    node.state = State.HOME
    node.set_mode_client.call_async = lambda req: _FakeSetModeFuture(
        _FakeSetModeResponse(success=False, message='camera busy'))

    node._set_vision_mode(SetVisionMode.Request.OFF)

    assert node.state == State.HOME
