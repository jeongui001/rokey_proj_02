import json

import rclpy
import pytest
from std_msgs.msg import String

from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode
from task_manager.command_parser import Mode
from task_manager.task_manager_node import Safety, State, TaskManagerNode


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


class _FakeSendGoalFuture:
    """robot_task_client.send_goal_async()가 반환하는 future를 흉내낸다."""

    def __init__(self):
        self._callback = None

    def add_done_callback(self, cb):
        self._callback = cb

    def fire(self, goal_handle):
        self._callback(_FakeFuture(goal_handle))


class _FakeResultFuture:
    """goal_handle.get_result_async()가 반환하는 future를 흉내낸다."""

    def __init__(self):
        self._callback = None

    def add_done_callback(self, cb):
        self._callback = cb

    def fire(self, result):
        self._callback(_FakeFuture(_FakeResponse(result)))


class _FakeGoalHandle:
    def __init__(self):
        self.accepted = True
        self.cancel_called = False
        self.result_future = _FakeResultFuture()

    def cancel_goal_async(self):
        self.cancel_called = True

    def get_result_async(self):
        return self.result_future


def _send_and_accept(node, task_type, **kwargs):
    """_send_robot_goal을 호출하고 goal이 즉시 수락된 것처럼 시뮬레이션해 goal_handle을 돌려준다."""
    send_future = _FakeSendGoalFuture()
    node.robot_task_client.send_goal_async = (
        lambda goal, feedback_callback=None: send_future)
    node._send_robot_goal(task_type, **kwargs)
    goal_handle = _FakeGoalHandle()
    send_future.fire(goal_handle)
    return goal_handle


# ---- 초기 상태 / 상태 JSON ----

def test_initial_state_is_idle_manual_and_normal(node):
    assert node.state == State.IDLE
    assert node.operation_mode == Mode.MANUAL
    assert node.safety_state == Safety.NORMAL


def test_set_state_publishes_json_status_with_mode_and_safety(node):
    published = []
    node.pub_status.publish = published.append

    node._set_state(State.PARSING, detail='hello')

    assert node.state == State.PARSING
    assert len(published) == 1
    payload = json.loads(published[0].data)
    assert payload == {
        'state': 'PARSING',
        'detail': 'hello',
        'operation_mode': Mode.MANUAL,
        'safety_state': Safety.NORMAL,
    }


# ---- /robot/fault 처리 ----

def test_fault_sets_fault_safety_state_and_requests_cancel_and_vision_off(node):
    node.state = State.SERVO_PICK
    vision_calls = []
    node._set_vision_mode = lambda mode, tool_class='': vision_calls.append(mode)
    goal_handle = _send_and_accept(node, 'servo_pick', tool_class='spanner')
    node.state = State.SERVO_PICK

    node._on_fault(String(data='torque anomaly'))

    assert node.safety_state == Safety.FAULT
    assert node.state == State.SERVO_PICK
    assert goal_handle.cancel_called
    assert SetVisionMode.Request.OFF in vision_calls
    # 취소 확인 전이므로 goal_in_progress는 아직 유지된다
    assert node._goal_in_progress is True


def test_fault_suppresses_stale_cancelled_result_dispatch(node):
    node.state = State.SERVO_PICK
    node._set_vision_mode = lambda mode, tool_class='': None
    goal_handle = _send_and_accept(node, 'servo_pick', tool_class='spanner')
    node.state = State.SERVO_PICK

    node._on_fault(String(data='torque anomaly'))
    # 취소된 goal의 결과(예: torque anomaly로 인한 실패)가 뒤늦게 도착해도
    # servo_pick 실패 분기를 다시 타서 상태를 바꾸면 안 된다.
    goal_handle.result_future.fire(_FakeResult(success=False, message='torque anomaly'))

    assert node.state == State.SERVO_PICK
    assert node.safety_state == Safety.FAULT
    assert node._goal_in_progress is False


def test_fault_with_protective_keyword_sets_protective_stop(node):
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_fault(String(data='protective stop triggered'))

    assert node.safety_state == Safety.PROTECTIVE_STOP


def test_fault_ignored_if_already_faulted(node):
    node.safety_state = Safety.FAULT
    published = []
    node.pub_status.publish = published.append

    node._on_fault(String(data='another fault'))

    assert published == []


# ---- 리셋: NORMAL로 자동 복귀하지 않음 ----

def test_reset_moves_to_recovery_required_not_normal(node):
    node.safety_state = Safety.FAULT
    node.state = State.SERVO_PICK

    node._on_user_command(String(data='리셋'))

    assert node.safety_state == Safety.RECOVERY_REQUIRED
    assert node.safety_state != Safety.NORMAL


def test_commands_still_blocked_after_recovery_required(node):
    node.safety_state = Safety.RECOVERY_REQUIRED
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='홈으로 가'))

    assert sent == []
    assert node.safety_state == Safety.RECOVERY_REQUIRED


def test_reset_again_from_recovery_required_is_noop(node):
    node.safety_state = Safety.RECOVERY_REQUIRED
    published = []
    node.pub_status.publish = published.append

    node._on_user_command(String(data='리셋'))

    assert node.safety_state == Safety.RECOVERY_REQUIRED
    assert len(published) == 1


def test_commands_ignored_while_faulted_except_reset(node):
    node.safety_state = Safety.FAULT
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='물병 갖다줘'))

    assert sent == []
    assert node.safety_state == Safety.FAULT


# ---- AUTO/MANUAL 모드 전환 (취소 확인 후 전환) ----

def test_mode_switch_immediate_when_idle_and_no_goal(node):
    node._on_user_command(String(data='자동 모드로 전환해줘'))

    assert node.operation_mode == Mode.AUTO
    assert node.state == State.IDLE


def test_mode_switch_requests_vision_off(node):
    calls = []
    node._set_vision_mode = lambda mode, tool_class='': calls.append(mode)

    node._on_user_command(String(data='자동 모드로 전환해줘'))

    assert SetVisionMode.Request.OFF in calls


def test_mode_switch_waits_for_cancel_confirmation_before_switching(node):
    node.operation_mode = Mode.AUTO
    node.state = State.SERVO_PICK
    goal_handle = _send_and_accept(node, 'servo_pick', tool_class='spanner')
    node.state = State.SERVO_PICK

    node._on_user_command(String(data='수동 모드로 전환해줘'))

    assert goal_handle.cancel_called
    assert node.operation_mode == Mode.AUTO  # 아직 전환되지 않음
    assert node.state == State.CANCELLING

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.operation_mode == Mode.MANUAL
    assert node.state == State.IDLE


# ---- STOP: AUTO/MANUAL 공통으로 취소 + Vision OFF ----

def test_stop_cancels_goal_during_manual_move(node):
    node.operation_mode = Mode.MANUAL
    goal_handle = _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.MANUAL_MOVE
    vision_calls = []
    node._set_vision_mode = lambda mode, tool_class='': vision_calls.append(mode)

    node._on_user_command(String(data='멈춰'))

    assert goal_handle.cancel_called
    assert SetVisionMode.Request.OFF in vision_calls
    assert node.state == State.CANCELLING
    assert node._goal_in_progress is True  # 취소 확인 전에는 아직 진행 중으로 본다

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.state == State.IDLE
    assert node._goal_in_progress is False


def test_stop_cancels_goal_during_auto_task(node):
    node.operation_mode = Mode.AUTO
    node.state = State.SERVO_PICK
    goal_handle = _send_and_accept(node, 'servo_pick', tool_class='spanner')
    node.state = State.SERVO_PICK
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_user_command(String(data='멈춰'))

    assert goal_handle.cancel_called
    assert node.state == State.CANCELLING

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.state == State.IDLE
    assert node._goal_in_progress is False


def test_stop_with_no_goal_in_progress_returns_to_idle_immediately(node):
    node.operation_mode = Mode.MANUAL
    node.state = State.DETECT_TRACK

    node._on_user_command(String(data='멈춰'))

    assert node.state == State.IDLE


# ---- 세대 번호로 지연 도착한 이전 goal 결과 무시 ----

def test_stale_goal_result_after_new_goal_is_ignored(node):
    node.operation_mode = Mode.MANUAL
    old_handle = _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.MANUAL_MOVE
    old_generation = node._goal_generation

    node._goal_in_progress = False  # 새 goal 전송을 허용하기 위한 시뮬레이션
    new_handle = _send_and_accept(node, 'move_named', named_target='front')
    node.state = State.MANUAL_MOVE

    assert node._goal_generation != old_generation

    old_handle.result_future.fire(_FakeResult(success=False, message='old stale result'))

    assert node._current_goal_handle is new_handle
    assert node._goal_in_progress is True
    assert node.state == State.MANUAL_MOVE


# ---- GoalHandle을 받기 전에 STOP/모드 전환/Fault가 들어오는 경우 ----

def test_stop_before_goal_accepted_stores_pending_cancel_and_cancels_on_accept(node):
    node.operation_mode = Mode.MANUAL
    send_future = _FakeSendGoalFuture()
    node.robot_task_client.send_goal_async = (
        lambda goal, feedback_callback=None: send_future)
    node._send_robot_goal('move_named', named_target='home')
    node.state = State.MANUAL_MOVE
    # 아직 GoalHandle을 받지 못한 상태(send_future.fire() 호출 전)에서 STOP이 들어온다.

    node._on_user_command(String(data='멈춰'))

    assert node.state == State.CANCELLING
    assert node._cancel_pending_callback is not None
    assert node._goal_in_progress is True  # 아직 취소 완료로 보지 않는다

    goal_handle = _FakeGoalHandle()
    send_future.fire(goal_handle)  # 뒤늦게 goal이 수락됨

    assert goal_handle.cancel_called is True  # 수락 즉시 취소가 걸린다
    assert node.state == State.CANCELLING  # result 도착 전까지는 아직 완료 아님

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.state == State.IDLE
    assert node._goal_in_progress is False


def test_stop_before_goal_rejected_cleans_up_pending_cancel_safely(node):
    node.operation_mode = Mode.MANUAL
    send_future = _FakeSendGoalFuture()
    node.robot_task_client.send_goal_async = (
        lambda goal, feedback_callback=None: send_future)
    node._send_robot_goal('move_named', named_target='home')
    node.state = State.MANUAL_MOVE

    node._on_user_command(String(data='멈춰'))
    assert node._cancel_pending_callback is not None

    rejected_handle = _FakeGoalHandle()
    rejected_handle.accepted = False
    send_future.fire(rejected_handle)

    # goal이 애초에 거절되었으므로 별도 fault로 취급하지 않고 취소 완료로 정리한다.
    assert node.state == State.IDLE
    assert node._goal_in_progress is False
    assert node._cancel_pending_callback is None
    assert node.safety_state == Safety.NORMAL


# ---- 취소 확인 타임아웃 ----

def test_cancel_timeout_sets_fault_and_suppresses_late_result(node):
    node.operation_mode = Mode.MANUAL
    goal_handle = _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.MANUAL_MOVE

    node._on_user_command(String(data='멈춰'))
    assert node._cancel_timeout_timer is not None

    node._on_cancel_timeout()

    assert node.safety_state == Safety.FAULT
    assert node._cancel_timeout_timer is None

    # 타임아웃 이후 뒤늦게 도착한 실제 result는 상태 전이에 반영되면 안 된다.
    goal_handle.result_future.fire(_FakeResult(success=True, message='late'))

    assert node.safety_state == Safety.FAULT
    assert node._goal_in_progress is False


def test_cancel_timeout_ignored_if_already_confirmed(node):
    node.operation_mode = Mode.MANUAL
    goal_handle = _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.MANUAL_MOVE

    node._on_user_command(String(data='멈춰'))
    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))
    assert node.state == State.IDLE

    # 이미 취소가 정상적으로 확인된 뒤에 타이머가 뒤늦게 남아있어도 안전해야 한다.
    node._on_cancel_timeout()

    assert node.safety_state == Safety.NORMAL
    assert node.state == State.IDLE


# ---- CANCELLING 중 추가 STOP/모드 전환이 기존 콜백을 덮어쓰지 않음 ----

def test_second_stop_while_cancelling_does_not_overwrite_pending_callback(node):
    node.operation_mode = Mode.MANUAL
    goal_handle = _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.MANUAL_MOVE

    node._on_user_command(String(data='멈춰'))
    first_callback = node._cancel_pending_callback
    assert first_callback is not None

    node._on_user_command(String(data='멈춰'))  # 두 번째 STOP

    assert node._cancel_pending_callback is first_callback

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.state == State.IDLE


def test_mode_switch_while_cancelling_does_not_overwrite_pending_stop_callback(node):
    node.operation_mode = Mode.AUTO
    node.state = State.SERVO_PICK
    goal_handle = _send_and_accept(node, 'servo_pick', tool_class='spanner')
    node.state = State.SERVO_PICK

    node._on_user_command(String(data='멈춰'))
    first_callback = node._cancel_pending_callback

    node._on_user_command(String(data='수동 모드로 전환해줘'))

    assert node._cancel_pending_callback is first_callback
    assert node.operation_mode == Mode.AUTO  # 모드 전환은 아직 반영되지 않음

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.state == State.IDLE  # 원래 STOP 콜백이 수행됨
    assert node.operation_mode == Mode.AUTO  # 모드 전환 요청은 무시되었으므로 재요청 필요


def test_fault_overrides_pending_stop_cancel_callback(node):
    node.operation_mode = Mode.MANUAL
    goal_handle = _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.MANUAL_MOVE
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_user_command(String(data='멈춰'))
    assert node.state == State.CANCELLING

    node._on_fault(String(data='torque anomaly'))
    assert node.safety_state == Safety.FAULT

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    # STOP이 의도했던 IDLE 전환이 아니라 fault로 인해 상태가 그대로 유지된다.
    assert node.state == State.CANCELLING
    assert node.safety_state == Safety.FAULT


# ---- MANUAL 이동 명령 ----

def test_manual_move_sends_named_target_goal(node):
    node.operation_mode = Mode.MANUAL
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='홈으로 가'))

    assert node.state == State.MANUAL_MOVE
    assert sent == [('move_named', {'named_target': 'home'})]


@pytest.mark.parametrize('text,named_target', [
    ('정면을 봐', 'front'),
    ('위를 봐', 'up'),
    ('아래를 봐', 'down'),
    ('컨베이어를 봐', 'watch'),
])
def test_manual_move_named_targets(node, text, named_target):
    node.operation_mode = Mode.MANUAL
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data=text))

    assert sent == [('move_named', {'named_target': named_target})]


def test_manual_move_ignored_in_auto_mode(node):
    node.operation_mode = Mode.AUTO
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='홈으로 가'))

    assert sent == []
    assert node.state == State.IDLE


def test_manual_move_result_success_returns_idle(node):
    node.state = State.MANUAL_MOVE

    node._on_robot_result(_FakeFuture(_FakeResponse(_FakeResult(success=True))), 0)

    assert node.state == State.IDLE


def test_manual_move_result_failure_sets_fault(node):
    node.state = State.MANUAL_MOVE
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_robot_result(
        _FakeFuture(_FakeResponse(_FakeResult(success=False, message='motion failed'))), 0)

    assert node.safety_state == Safety.FAULT


# ---- 중복 goal 방지 ----

def test_duplicate_goal_is_ignored_while_in_progress(node):
    called = []
    node.robot_task_client.send_goal_async = lambda goal, feedback_callback=None: called.append(goal)
    node._goal_in_progress = True

    node._send_robot_goal('move_named', named_target='home')

    assert called == []


def test_goal_send_sets_in_progress_flag_and_bumps_generation(node):
    generation_before = node._goal_generation

    class _FakeSendFuture:
        def add_done_callback(self, cb):
            pass

    node.robot_task_client.send_goal_async = lambda goal, feedback_callback=None: _FakeSendFuture()

    node._send_robot_goal('move_named', named_target='home')

    assert node._goal_in_progress is True
    assert node._goal_generation == generation_before + 1


# ---- AUTO 공구 전달 명령 (기존 상태머신 배선 유지) ----

def test_user_command_ignored_unless_idle(node):
    node.operation_mode = Mode.AUTO
    node.state = State.SERVO_PICK
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='스패너 갖다줘'))

    assert sent == []


def test_fetch_tool_ignored_in_manual_mode(node):
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='스패너 갖다줘'))

    assert sent == []
    assert node.current_tool is None


def test_fetch_tool_requires_auto_mode_switch_first(node):
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='스패너 갖다줘'))
    assert sent == []

    node._on_user_command(String(data='자동 모드로 전환해줘'))
    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_user_command(String(data='스패너 갖다줘'))

    assert node.state == State.MOVE_TO_WATCH
    assert sent == [('move_named', {'named_target': 'watch'})]


def test_user_command_triggers_move_to_watch(node):
    node.operation_mode = Mode.AUTO
    node.state = State.IDLE
    sent_goals = []
    node._send_robot_goal = lambda task_type, **kw: sent_goals.append((task_type, kw))
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_user_command(String(data='스패너 갖다줘'))

    assert node.state == State.MOVE_TO_WATCH
    assert node.current_tool == 'spanner'
    assert sent_goals == [('move_named', {'named_target': 'watch'})]


def test_unknown_command_reports_and_stays_idle(node):
    node.state = State.IDLE

    node._on_user_command(String(data='asdf'))

    assert node.state == State.IDLE


def test_move_to_watch_result_success_transitions_to_detect_track(node):
    node.state = State.MOVE_TO_WATCH

    node._handle_move_to_watch_result(_FakeResult(success=True))

    assert node.state == State.DETECT_TRACK
    assert node._detect_track_timer is not None
    node._detect_track_timer.cancel()


def test_move_to_watch_result_failure_sets_fault(node):
    node.state = State.MOVE_TO_WATCH
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_move_to_watch_result(_FakeResult(success=False, message='motion failed'))

    assert node.safety_state == Safety.FAULT
    assert node.state == State.MOVE_TO_WATCH


# ---- DETECT_TRACK: 초 단위 타임아웃 ----

def test_detect_track_timeout_requests_vision_off_clears_tool_and_returns_idle(node):
    node.state = State.DETECT_TRACK
    node.current_tool = 'spanner'
    node._detect_track_timer = node.create_timer(100.0, lambda: None)
    vision_calls = []
    node._set_vision_mode = lambda mode, tool_class='': vision_calls.append(mode)

    node._on_detect_track_timeout()

    assert node.state == State.IDLE
    assert node.current_tool is None
    assert SetVisionMode.Request.OFF in vision_calls


def test_detect_track_timeout_ignored_if_state_changed(node):
    node.state = State.SERVO_PICK

    node._on_detect_track_timeout()

    assert node.state == State.SERVO_PICK


def test_tool_track_ignored_unless_detect_track(node):
    node.state = State.IDLE
    node._check_trigger = lambda msg: True
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_tool_track(ToolTrack())

    assert sent == []


def test_tool_track_trigger_sends_servo_pick_goal_and_cancels_timer(node):
    node.state = State.DETECT_TRACK
    node.current_tool = 'spanner'
    node._check_trigger = lambda msg: True
    node._get_grasp_spec = lambda tool_class: (30.0, 20.0)
    node._detect_track_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_tool_track(ToolTrack())

    assert node.state == State.SERVO_PICK
    assert node._detect_track_timer is None
    assert sent == [('servo_pick', {
        'tool_class': 'spanner', 'grasp_width_mm': 30.0, 'grasp_force_n': 20.0})]


def test_tool_track_no_trigger_does_not_change_state(node):
    node.state = State.DETECT_TRACK
    node._check_trigger = lambda msg: False

    node._on_tool_track(ToolTrack())

    assert node.state == State.DETECT_TRACK


# ---- SERVO_PICK / VERIFY_GRASP / MOVE_SAFE / WAIT_PULL / RELEASE / HOME ----

def test_servo_pick_result_torque_anomaly_sets_fault(node):
    node.state = State.SERVO_PICK
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_servo_pick_result(_FakeResult(success=False, message='torque anomaly'))

    assert node.safety_state == Safety.FAULT
    assert node.state == State.SERVO_PICK


def test_servo_pick_result_other_failure_returns_to_detect_track_with_timer(node):
    node.state = State.SERVO_PICK

    node._handle_servo_pick_result(_FakeResult(success=False, message='timeout'))

    assert node.state == State.DETECT_TRACK
    assert node._detect_track_timer is not None
    node._detect_track_timer.cancel()


def test_servo_pick_result_success_and_verify_passes_moves_to_move_safe_with_handover_safe(node):
    node.state = State.SERVO_PICK
    node._verify_grasp = lambda result: True
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_servo_pick_result(_FakeResult(success=True))

    assert node.state == State.MOVE_SAFE
    assert sent == [('move_named', {'named_target': 'handover_safe'})]


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


def test_verify_grasp_exceeds_max_retries_enters_fault_without_opening_gripper(node):
    node.state = State.SERVO_PICK
    node._verify_grasp = lambda result: False
    node._verify_grasp_retries = 2
    vision_calls = []
    node._set_vision_mode = lambda mode, tool_class='': vision_calls.append(mode)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_servo_pick_result(_FakeResult(success=True))

    assert node.safety_state == Safety.FAULT
    assert node.state == State.VERIFY_GRASP  # 실패 지점 그대로 보존
    assert SetVisionMode.Request.OFF in vision_calls
    assert sent == []  # release_and_retry 등 그리퍼를 여는 goal을 보내지 않는다


def test_release_and_retry_result_success_returns_to_detect_track(node):
    node.state = State.VERIFY_GRASP

    node._handle_release_and_retry_result(_FakeResult(success=True))

    assert node.state == State.DETECT_TRACK
    node._detect_track_timer.cancel()


def test_release_and_retry_result_failure_sets_fault(node):
    node.state = State.VERIFY_GRASP
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_release_and_retry_result(_FakeResult(success=False, message='release failed'))

    assert node.safety_state == Safety.FAULT


def test_move_safe_result_success_transitions_directly_to_wait_pull(node):
    node.state = State.MOVE_SAFE
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_move_safe_result(_FakeResult(success=True))

    assert node.state == State.WAIT_PULL
    assert sent == [('handover_hold', {})]
    assert node._wait_pull_timeout_timer is not None
    node._wait_pull_timeout_timer.cancel()


def test_move_safe_result_failure_sets_fault(node):
    node.state = State.MOVE_SAFE
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_move_safe_result(_FakeResult(success=False, message='motion failed'))

    assert node.safety_state == Safety.FAULT


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


def test_wait_pull_timeout_cancels_handover_hold_before_sending_place_down(node):
    node.state = State.WAIT_PULL
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)
    handover_hold_handle = _send_and_accept(node, 'handover_hold')
    node.state = State.WAIT_PULL

    node._on_wait_pull_timeout()

    assert handover_hold_handle.cancel_called
    assert node.state == State.WAIT_PULL  # 취소 확인 전에는 아직 RELEASE로 전이하지 않는다
    assert node._goal_in_progress is True

    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))
    handover_hold_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.state == State.RELEASE
    # place_down이 중복 goal 방지 가드에 의해 무시되지 않고 실제로 전송된다
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


def test_home_result_failure_sets_fault(node):
    node.state = State.HOME
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_home_result(_FakeResult(success=False, message='motion failed'))

    assert node.safety_state == Safety.FAULT
