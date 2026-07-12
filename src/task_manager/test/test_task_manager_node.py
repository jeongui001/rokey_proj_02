import json

import rclpy
import pytest
from rclpy.parameter import Parameter
from std_msgs.msg import String

from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode
from task_manager.command_parser import Mode
from task_manager.task_manager_node import (
    WAIT_PULL_REMINDER_MESSAGE, GraspSpec, Safety, State, TaskManagerNode,
)


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


class _FakeTriggerResponse:
    def __init__(self, success=True, message=''):
        self.success = success
        self.message = message


class _FakeRecoverFuture:
    """recover_client.call_async()가 반환하는 future를 흉내낸다."""

    def __init__(self):
        self._callback = None
        self._result = None
        self._exception = None

    def add_done_callback(self, cb):
        self._callback = cb

    def fire(self, response):
        self._result = response
        self._callback(self)

    def fire_exception(self, exc):
        self._exception = exc
        self._callback(self)

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


class _FakeVisionFuture:
    def __init__(self):
        self._callback = None
        self._response = None

    def add_done_callback(self, callback):
        self._callback = callback

    def result(self):
        return self._response

    def fire(self, success=True, message=''):
        self._response = SetVisionMode.Response(success=success, message=message)
        self._callback(self)


class _FakeRecoverClient:
    """task_manager.recover_client(std_srvs/Trigger 클라이언트)를 흉내낸다."""

    def __init__(self, ready=True):
        self._ready = ready
        self.call_count = 0
        self.last_future = None

    def service_is_ready(self):
        return self._ready

    def call_async(self, request):
        self.call_count += 1
        self.last_future = _FakeRecoverFuture()
        return self.last_future


# ---- 초기 상태 / 상태 JSON ----

def test_initial_state_is_idle_manual_and_normal(node):
    assert node.state == State.IDLE
    assert node.operation_mode == Mode.MANUAL
    assert node.safety_state == Safety.NORMAL


def test_set_state_publishes_json_status_with_mode_and_safety(node):
    published = []
    node.pub_status.publish = published.append

    node._set_state(State.MOVE_TO_WATCH, detail='hello')

    assert node.state == State.MOVE_TO_WATCH
    assert len(published) == 1
    payload = json.loads(published[0].data)
    assert payload == {
        'state': 'MOVE_TO_WATCH',
        'detail': 'hello',
        'operation_mode': Mode.MANUAL,
        'safety_state': Safety.NORMAL,
        'resumable': False,
    }


# ---- 초기/주기적 /task/status 발행 (늦게 연결된 GUI 포함) ----

def test_initial_status_published_on_init_is_idle_manual_normal(node):
    # __init__ 안에서 이미 한 번 발행되었다(초기 연결된 GUI를 위함) - 그 부작용으로
    # _last_status_detail이 빈 문자열로 설정되어 있어야 한다.
    assert node._last_status_detail == ''
    assert node.state == State.IDLE
    assert node.operation_mode == Mode.MANUAL
    assert node.safety_state == Safety.NORMAL


def test_status_publish_timer_republishes_full_four_fields(node):
    published = []
    node.pub_status.publish = published.append

    node._on_status_publish_timer()

    assert len(published) == 1
    payload = json.loads(published[0].data)
    assert set(payload.keys()) == {
        'state', 'detail', 'operation_mode', 'safety_state', 'resumable'}
    assert payload == {
        'state': State.IDLE, 'detail': '', 'operation_mode': Mode.MANUAL,
        'safety_state': Safety.NORMAL, 'resumable': False,
    }


def test_status_publish_timer_keeps_last_detail(node):
    node._set_state(State.MOVE_TO_WATCH, detail='hello')
    published = []
    node.pub_status.publish = published.append

    node._on_status_publish_timer()

    payload = json.loads(published[0].data)
    assert payload['detail'] == 'hello'
    assert payload['state'] == State.MOVE_TO_WATCH


def test_status_publish_timer_does_not_change_state_or_detail(node):
    node._set_state(State.DETECT_TRACK, detail='watching')

    node._on_status_publish_timer()
    node._on_status_publish_timer()

    assert node.state == State.DETECT_TRACK
    assert node._last_status_detail == 'watching'


def test_status_publish_timer_republishes_fault_state(node):
    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='FAULT: torque anomaly'))
    published = []
    node.pub_status.publish = published.append

    node._on_status_publish_timer()

    payload = json.loads(published[0].data)
    assert payload['safety_state'] == Safety.FAULT


def test_status_publish_timer_reflects_recovery_completed_state(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client
    node._on_user_command(String(data='리셋'))
    fake_client.last_future.fire(_FakeTriggerResponse(success=True, message='복구됨'))
    assert node.safety_state == Safety.NORMAL
    published = []
    node.pub_status.publish = published.append

    node._on_status_publish_timer()

    payload = json.loads(published[0].data)
    assert payload['safety_state'] == Safety.NORMAL
    assert payload['state'] == State.IDLE


def test_cancel_all_timers_does_not_cancel_status_timer(node):
    node._cancel_all_timers()

    assert node._status_publish_timer is not None


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


@pytest.mark.parametrize('message,expected', [
    ('PROTECTIVE_STOP: torque anomaly', Safety.PROTECTIVE_STOP),
    ('EMERGENCY_STOP: e-stop pressed', Safety.EMERGENCY_STOP),
    ('FAULT: unexpected force', Safety.FAULT),
    ('unknown format without prefix', Safety.FAULT),
    # 단순 부분 문자열 포함(예: 소문자 'protective')은 더 이상 인정하지 않고, 정확한
    # 접두어가 없으므로 안전하게 FAULT로 분류한다.
    ('protective stop triggered', Safety.FAULT),
])
def test_fault_prefix_classification(node, message, expected):
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_fault(String(data=message))

    assert node.safety_state == expected


def test_fault_ignores_exact_duplicate_message_at_same_grade(node):
    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='FAULT: torque anomaly'))
    published = []
    node.pub_status.publish = published.append

    node._on_fault(String(data='FAULT: torque anomaly'))  # 완전히 동일한 메시지 반복

    assert published == []
    assert node.safety_state == Safety.FAULT


def test_fault_same_grade_different_message_is_not_ignored(node):
    # 이미 비정상 상태라는 이유만으로 모든 새 Fault를 무시하지는 않는다 - 같은 등급
    # 이라도 메시지(detail)가 다르면 최신 정보를 반영한다.
    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='FAULT: torque anomaly'))
    published = []
    node.pub_status.publish = published.append

    node._on_fault(String(data='FAULT: another distinct fault'))

    assert len(published) == 1
    assert node.safety_state == Safety.FAULT


def test_new_fault_during_recovery_required_updates_safety_state(node):
    node.safety_state = Safety.RECOVERY_REQUIRED
    node._recovery_in_progress = True  # 복구 시도가 진행 중이라고 가정
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_fault(String(data='PROTECTIVE_STOP: new issue during recovery'))

    assert node.safety_state == Safety.PROTECTIVE_STOP
    assert node._recovery_in_progress is False  # 새 Fault가 진행 중이던 복구 시도를 무효화


# ---- Fault 단계 상승(escalation) 우선순위 ----

def test_fault_protective_stop_then_emergency_stop_escalates(node):
    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='PROTECTIVE_STOP: bumper contact'))
    assert node.safety_state == Safety.PROTECTIVE_STOP

    node._on_fault(String(data='EMERGENCY_STOP: e-stop pressed'))

    assert node.safety_state == Safety.EMERGENCY_STOP


def test_fault_fault_then_emergency_stop_escalates(node):
    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='FAULT: torque anomaly'))
    assert node.safety_state == Safety.FAULT

    node._on_fault(String(data='EMERGENCY_STOP: e-stop pressed'))

    assert node.safety_state == Safety.EMERGENCY_STOP


def test_fault_emergency_stop_is_never_downgraded_by_lower_fault(node):
    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='EMERGENCY_STOP: e-stop pressed'))
    assert node.safety_state == Safety.EMERGENCY_STOP
    published = []
    node.pub_status.publish = published.append

    node._on_fault(String(data='PROTECTIVE_STOP: bumper contact'))

    assert node.safety_state == Safety.EMERGENCY_STOP  # 강등되지 않는다
    assert published == []


# ---- 리셋: /robot/recover 연결, RECOVERY_REQUIRED 경유 ----

def test_reset_moves_to_recovery_required_not_normal(node):
    node.safety_state = Safety.FAULT
    node.state = State.SERVO_PICK
    node.recover_client = _FakeRecoverClient(ready=False)

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


def test_commands_ignored_while_faulted_except_reset(node):
    node.safety_state = Safety.FAULT
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='물병 갖다줘'))

    assert sent == []
    assert node.safety_state == Safety.FAULT


def test_recover_success_true_sets_normal_manual_idle(node):
    node.safety_state = Safety.FAULT
    node.state = State.SERVO_PICK
    node.current_tool = 'spanner'
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    assert node.safety_state == Safety.RECOVERY_REQUIRED
    assert fake_client.call_count == 1

    fake_client.last_future.fire(_FakeTriggerResponse(success=True, message='복구됨'))

    assert node.safety_state == Safety.NORMAL
    assert node.operation_mode == Mode.MANUAL
    assert node.state == State.IDLE
    assert node.current_tool is None


def test_recover_success_false_keeps_recovery_required(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    fake_client.last_future.fire(_FakeTriggerResponse(success=False, message='아직 안전하지 않음'))

    assert node.safety_state == Safety.RECOVERY_REQUIRED


def test_recover_service_not_ready_keeps_recovery_required_and_allows_retry(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=False)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))

    assert node.safety_state == Safety.RECOVERY_REQUIRED
    assert fake_client.call_count == 0  # 서비스가 준비되지 않아 호출 자체를 하지 않는다
    assert node._recovery_in_progress is False  # 재시도가 가능해야 한다

    fake_client._ready = True
    node._on_user_command(String(data='리셋'))

    assert fake_client.call_count == 1


def test_recover_future_exception_keeps_recovery_required(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    fake_client.last_future.fire_exception(RuntimeError('boom'))

    assert node.safety_state == Safety.RECOVERY_REQUIRED
    assert node._recovery_in_progress is False


def test_recover_call_async_exception_keeps_recovery_required_and_allows_retry(node):
    node.safety_state = Safety.FAULT

    class _RaisingRecoverClient:
        def service_is_ready(self):
            return True

        def call_async(self, request):
            raise RuntimeError('recover request boom')

    node.recover_client = _RaisingRecoverClient()
    node._on_user_command(String(data='리셋'))

    assert node.safety_state == Safety.RECOVERY_REQUIRED
    assert node._recovery_in_progress is False
    assert node._recovery_timeout_timer is None


def test_duplicate_reset_while_recovery_in_progress_does_not_call_service_twice(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    published = []
    node.pub_status.publish = published.append

    node._on_user_command(String(data='리셋'))  # 아직 응답 전 - 중복 요청

    assert node.safety_state == Safety.RECOVERY_REQUIRED
    assert len(published) == 1
    assert fake_client.call_count == 1  # 서비스가 다시 호출되지 않는다


def test_recover_waits_for_goal_cancel_before_calling_service(node):
    node.operation_mode = Mode.AUTO
    node.state = State.SERVO_PICK
    goal_handle = _send_and_accept(node, 'servo_pick', tool_class='spanner')
    node.state = State.SERVO_PICK
    node.safety_state = Safety.FAULT  # goal은 아직 취소 대기 중이라고 가정
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))

    assert fake_client.call_count == 0  # 아직 goal 취소가 확인되지 않았다
    assert goal_handle.cancel_called

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert fake_client.call_count == 1  # 취소가 확인된 뒤에야 호출된다


def test_recover_not_called_after_cancel_timeout(node):
    node.operation_mode = Mode.MANUAL
    goal_handle = _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.MANUAL_MOVE
    node.safety_state = Safety.FAULT  # 리셋이 동작하려면 이미 비정상 상태여야 한다
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    assert node._cancel_timeout_timer is not None

    node._on_cancel_timeout()

    assert node.safety_state == Safety.FAULT
    assert node._recovery_in_progress is False
    assert fake_client.call_count == 0

    # 타임아웃 이후 뒤늦게 도착한 result도 recover를 호출하면 안 된다.
    goal_handle.result_future.fire(_FakeResult(success=True, message='late'))

    assert fake_client.call_count == 0


def test_stale_recover_success_ignored_after_new_fault(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    pending_future = fake_client.last_future
    assert node.safety_state == Safety.RECOVERY_REQUIRED

    # 응답이 오기 전에 새로운(더 심각한) Fault가 발생한다.
    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='EMERGENCY_STOP: e-stop pressed'))
    assert node.safety_state == Safety.EMERGENCY_STOP

    # 오래된 복구 요청의 success 응답이 뒤늦게 도착해도 무시되어야 한다.
    pending_future.fire(_FakeTriggerResponse(success=True, message='늦은 성공'))

    assert node.safety_state == Safety.EMERGENCY_STOP  # NORMAL로 덮어써지지 않는다


# ---- /robot/recover 응답 타임아웃 ----

def test_recover_success_before_timeout_cancels_timeout_timer(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    assert node._recovery_timeout_timer is not None

    fake_client.last_future.fire(_FakeTriggerResponse(success=True, message='복구됨'))

    assert node._recovery_timeout_timer is None
    assert node.safety_state == Safety.NORMAL


def test_recover_timeout_when_no_response_keeps_recovery_required(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    generation = node._recovery_generation
    assert node._recovery_timeout_timer is not None

    node._on_recovery_timeout(generation)

    assert node.safety_state == Safety.RECOVERY_REQUIRED  # 자동으로 NORMAL 전환하지 않는다
    assert node._recovery_in_progress is False
    assert node._recovery_timeout_timer is None


def test_recover_timeout_allows_retry(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    node._on_recovery_timeout(node._recovery_generation)
    assert fake_client.call_count == 1

    node._on_user_command(String(data='리셋'))  # 타임아웃 이후 재시도

    assert fake_client.call_count == 2
    assert node.safety_state == Safety.RECOVERY_REQUIRED


def test_recover_late_success_after_timeout_is_ignored(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    pending_future = fake_client.last_future
    node._on_recovery_timeout(node._recovery_generation)

    # 타임아웃으로 세대가 이미 올라간 뒤에 늦게 도착한 success 응답은 무시되어야 한다.
    pending_future.fire(_FakeTriggerResponse(success=True, message='늦은 성공'))

    assert node.safety_state == Safety.RECOVERY_REQUIRED
    assert node._recovery_in_progress is False


def test_recover_new_fault_after_timeout_late_success_still_ignored(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    pending_future = fake_client.last_future
    node._on_recovery_timeout(node._recovery_generation)

    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='EMERGENCY_STOP: e-stop pressed'))

    pending_future.fire(_FakeTriggerResponse(success=True, message='늦은 성공'))

    assert node.safety_state == Safety.EMERGENCY_STOP


def test_new_fault_stops_recovery_timeout_timer(node):
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    assert node._recovery_timeout_timer is not None

    node._set_vision_mode = lambda mode, tool_class='': None
    node._on_fault(String(data='EMERGENCY_STOP: e-stop pressed'))

    assert node._recovery_timeout_timer is None


def test_recovery_timeout_stale_generation_is_noop(node):
    # 이미 취소/처리가 끝난 뒤(다음 리셋으로 새 세대가 시작된 뒤)에 이전 세대의
    # 타임아웃 콜백이 뒤늦게 실행돼도 아무 영향이 없어야 한다.
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    stale_generation = node._recovery_generation
    fake_client.last_future.fire(_FakeTriggerResponse(success=True, message='복구됨'))
    assert node.safety_state == Safety.NORMAL

    node._on_recovery_timeout(stale_generation)

    assert node.safety_state == Safety.NORMAL


def test_stale_recover_response_does_not_cancel_new_recovery_timer(node):
    # 이전 요청이 타임아웃된 뒤 재시도로 새 타이머가 만들어진 상태에서, 그 이전
    # 요청의 응답이 뒤늦게 도착해도 새 타이머를 취소하면 안 된다(generation 확인이
    # 먼저이며, 타이머 소유권도 별도로 확인한다).
    node.safety_state = Safety.FAULT
    fake_client = _FakeRecoverClient(ready=True)
    node.recover_client = fake_client

    node._on_user_command(String(data='리셋'))
    stale_future = fake_client.last_future
    stale_generation = node._recovery_generation
    node._on_recovery_timeout(stale_generation)  # 첫 요청 타임아웃 - generation 증가

    node._on_user_command(String(data='리셋'))  # 재시도 - 새 타이머 생성
    assert fake_client.call_count == 2
    new_timer = node._recovery_timeout_timer
    assert new_timer is not None

    # 오래된 첫 번째 요청의 응답이 뒤늦게 도착한다.
    stale_future.fire(_FakeTriggerResponse(success=True, message='늦은 성공'))

    assert node._recovery_timeout_timer is new_timer  # 새 타이머가 취소되지 않았다
    assert node.safety_state == Safety.RECOVERY_REQUIRED  # 늦은 성공으로 NORMAL 전환되지 않음


# ---- Action future 예외 처리 (send_goal_async/cancel_goal_async/future.result()) ----

def test_goal_response_future_exception_sets_fault_and_clears_flags(node):
    node.operation_mode = Mode.MANUAL
    send_future = _FakeSendGoalFuture()
    node.robot_task_client.send_goal_async = (
        lambda goal, feedback_callback=None: send_future)
    node._send_robot_goal('move_named', named_target='home')
    node.state = State.MANUAL_MOVE
    node._set_vision_mode = lambda mode, tool_class='': None
    generation = node._goal_generation

    class _RaisingFuture:
        def result(self):
            raise RuntimeError('goal response boom')

    node._on_goal_response(_RaisingFuture(), generation)

    assert node.safety_state == Safety.FAULT
    assert node._goal_in_progress is False
    assert node._current_goal_handle is None
    assert node._cancel_pending_callback is None
    assert node._goal_generation != generation  # 지연 콜백 무효화


def test_result_future_exception_sets_fault_and_clears_flags(node):
    node.operation_mode = Mode.MANUAL
    _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.MANUAL_MOVE
    node._set_vision_mode = lambda mode, tool_class='': None
    generation = node._goal_generation

    class _RaisingFuture:
        def result(self):
            raise RuntimeError('result boom')

    node._on_robot_result(_RaisingFuture(), generation)

    assert node.safety_state == Safety.FAULT
    assert node._goal_in_progress is False
    assert node._current_goal_handle is None
    assert node._goal_generation != generation  # 지연 콜백이 상태를 되돌리지 못하게 세대 무효화

    # 오래된 generation으로 다시 호출돼도(지연 재실행 등) 더 이상 아무 영향이 없다.
    node.state = State.HOME
    node._on_robot_result(_RaisingFuture(), generation)
    assert node.state == State.HOME


def test_result_future_exception_during_cancel_does_not_run_pending_callback(node):
    _send_and_accept(node, 'move_named', named_target='home')
    node.state = State.CANCELLING
    node._set_vision_mode = lambda mode, tool_class='': None
    callback_calls = []
    node._cancel_pending_callback = lambda: callback_calls.append(True)
    generation = node._goal_generation

    class _RaisingFuture:
        def result(self):
            raise RuntimeError('result during cancel boom')

    node._on_robot_result(_RaisingFuture(), generation)

    assert callback_calls == []
    assert node.safety_state == Safety.FAULT
    assert node._cancel_pending_callback is None
    assert node._goal_generation != generation


def test_get_result_async_exception_sets_fault_and_clears_flags(node):
    node._goal_in_progress = True
    node._goal_generation += 1
    generation = node._goal_generation
    node._set_vision_mode = lambda mode, tool_class='': None

    class _RaisingGoalHandle:
        accepted = True

        def get_result_async(self):
            raise RuntimeError('get result boom')

    node._on_goal_response(_FakeFuture(_RaisingGoalHandle()), generation)

    assert node.safety_state == Safety.FAULT
    assert node._goal_in_progress is False
    assert node._current_goal_handle is None
    assert node._goal_generation != generation


# ---- NaN/Inf confidence와 grasp spec 값 거부 ----

@pytest.mark.parametrize('case', [
    'confidence_nan', 'confidence_inf', 'confidence_out_of_range',
    'grasp_spec_width_nan', 'grasp_spec_force_inf',
])
def test_rejects_nan_inf_and_partial_config(node, case):
    if case == 'confidence_nan':
        _configure_trigger_params(node)
        node.current_tool = 'spanner'
        assert node._check_trigger(_fresh_tool_track(node, confidence=float('nan'))) is False
    elif case == 'confidence_inf':
        _configure_trigger_params(node)
        node.current_tool = 'spanner'
        assert node._check_trigger(_fresh_tool_track(node, confidence=float('inf'))) is False
    elif case == 'confidence_out_of_range':
        _configure_trigger_params(node)
        node.current_tool = 'spanner'
        assert node._check_trigger(_fresh_tool_track(node, confidence=1.5)) is False
    elif case == 'grasp_spec_width_nan':
        _configure_tool_spec(node, 'hammer', **{'tools.hammer.width_mm': float('nan')})
        assert node._get_grasp_spec('hammer') is None
    elif case == 'grasp_spec_force_inf':
        _configure_tool_spec(node, 'hammer', **{'tools.hammer.force_n': float('inf')})
        assert node._get_grasp_spec('hammer') is None
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


def test_fetch_tool_works_in_manual_mode_without_switching_to_auto(node):
    # fetch_tool은 AUTO/MANUAL 모드 구분 없이 동작한다 - config_ready만 갖춰지면
    # 굳이 자동 모드로 전환하지 않아도 타이핑/음성 명령만으로 전체 시퀀스가 시작된다.
    node.set_parameters([Parameter('auto.config_ready', value=True)])
    assert node.operation_mode == Mode.MANUAL  # 기본값 그대로
    node._set_vision_mode = lambda mode, tool_class='': None
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='망치 갖다줘'))

    assert node.state == State.MOVE_TO_WATCH
    assert sent == [('move_named', {'named_target': 'watch'})]


def test_user_command_triggers_move_to_watch(node):
    node.set_parameters([Parameter('auto.config_ready', value=True)])
    node.operation_mode = Mode.AUTO
    node.state = State.IDLE
    sent_goals = []
    node._send_robot_goal = lambda task_type, **kw: sent_goals.append((task_type, kw))
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_user_command(String(data='망치 갖다줘'))

    assert node.state == State.MOVE_TO_WATCH
    assert node.current_tool == 'hammer'
    assert sent_goals == [('move_named', {'named_target': 'watch'})]


def test_fetch_waits_for_vision_mode_success_before_robot_goal(node):
    node.set_parameters([Parameter('auto.config_ready', value=True)])
    node.operation_mode = Mode.AUTO
    future = _FakeVisionFuture()
    node.set_mode_client.service_is_ready = lambda: True
    node.set_mode_client.call_async = lambda request: future
    sent = []
    node._send_robot_goal = lambda task_type, **kwargs: sent.append((task_type, kwargs))

    node._handle_fetch_tool('spanner')

    assert sent == []
    future.fire(success=True)
    assert sent == [('move_named', {'named_target': 'watch'})]


def test_fetch_stops_when_vision_mode_change_fails(node):
    node.set_parameters([Parameter('auto.config_ready', value=True)])
    node.operation_mode = Mode.AUTO
    future = _FakeVisionFuture()
    node.set_mode_client.service_is_ready = lambda: True
    node.set_mode_client.call_async = lambda request: future
    node._send_robot_goal = lambda *args, **kwargs: pytest.fail('goal must not be sent')
    node._request_cancel = lambda callback: callback()

    node._handle_fetch_tool('spanner')
    future.fire(success=False, message='camera unavailable')

    assert node.safety_state == Safety.FAULT


def test_old_vision_response_cannot_start_goal_after_stop(node):
    node.set_parameters([Parameter('auto.config_ready', value=True)])
    node.operation_mode = Mode.AUTO
    future = _FakeVisionFuture()
    node.set_mode_client.service_is_ready = lambda: True
    node.set_mode_client.call_async = lambda request: future
    sent = []
    node._send_robot_goal = lambda task_type, **kwargs: sent.append((task_type, kwargs))

    node._handle_fetch_tool('spanner')
    node._set_vision_mode(SetVisionMode.Request.OFF)
    future.fire(success=True)

    assert sent == []


# ---- AUTO 설정 준비 게이트 (auto.config_ready) ----

def test_fetch_tool_blocked_when_config_not_ready(node):
    # auto.config_ready 기본값은 false다 - AUTO 모드 전환은 되지만 실제 goal은 보내지 않는다.
    node.operation_mode = Mode.AUTO
    node.state = State.IDLE
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_user_command(String(data='스패너 갖다줘'))

    assert sent == []
    assert node.state == State.IDLE
    assert node.current_tool is None


def test_mode_switch_to_auto_allowed_even_when_config_not_ready(node):
    node._on_user_command(String(data='자동 모드로 전환해줘'))

    assert node.operation_mode == Mode.AUTO  # 모드 전환 자체는 허용됨


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
    node._get_grasp_spec = lambda tool_class: GraspSpec(
        width_mm=30.0, force_n=20.0,
        verify_min_width_mm=25.0, verify_max_width_mm=35.0)
    node._detect_track_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_tool_track(ToolTrack())

    assert node.state == State.SERVO_PICK
    assert node._detect_track_timer is None
    assert sent == [('servo_pick', {
        'tool_class': 'spanner', 'grasp_width_mm': 30.0, 'grasp_force_n': 20.0})]
    assert node._active_grasp_spec.width_mm == 30.0


def test_tool_track_missing_grasp_spec_does_not_send_goal_and_returns_idle(node):
    node.state = State.DETECT_TRACK
    node.current_tool = 'spanner'
    node._check_trigger = lambda msg: True
    node._get_grasp_spec = lambda tool_class: None  # 미설정/유효하지 않음
    node._detect_track_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))
    vision_calls = []
    node._set_vision_mode = lambda mode, tool_class='': vision_calls.append(mode)

    node._on_tool_track(ToolTrack())

    assert sent == []  # servo_pick goal을 보내지 않는다
    assert node.state == State.IDLE
    assert node.current_tool is None
    assert node._active_grasp_spec is None
    assert SetVisionMode.Request.OFF in vision_calls


def test_tool_track_no_trigger_does_not_change_state(node):
    node.state = State.DETECT_TRACK
    node._check_trigger = lambda msg: False

    node._on_tool_track(ToolTrack())

    assert node.state == State.DETECT_TRACK


# ---- _check_trigger 실제 구현 ----

def _configure_trigger_params(node, **overrides):
    values = {
        'trigger.min_confidence': 0.8,
        'trigger.require_depth_valid': True,
        'trigger.require_approaching': True,
        'trigger.required_frame_id': 'base_link',
        'trigger.max_track_age_s': 1.0,
    }
    values.update(overrides)
    node.set_parameters([Parameter(name, value=value) for name, value in values.items()])


def _fresh_tool_track(node, tool_class='spanner', confidence=0.9, depth_valid=True,
                       approaching=True, frame_id='base_link', age_s=0.0,
                       x=0.1, y=0.1, z=0.1):
    msg = ToolTrack()
    msg.tool_class = tool_class
    msg.confidence = confidence
    msg.depth_valid = depth_valid
    msg.approaching = approaching
    msg.header.frame_id = frame_id
    stamp = node.get_clock().now() - rclpy.duration.Duration(seconds=age_s)
    msg.header.stamp = stamp.to_msg()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    return msg


def test_check_trigger_accepts_correct_tool(node):
    _configure_trigger_params(node)
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node, tool_class='spanner')) is True


def test_check_trigger_rejects_different_tool(node):
    _configure_trigger_params(node)
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node, tool_class='screwdriver')) is False


def test_check_trigger_rejects_no_current_tool(node):
    _configure_trigger_params(node)
    node.current_tool = None

    assert node._check_trigger(_fresh_tool_track(node, tool_class='spanner')) is False


def test_check_trigger_rejects_low_confidence(node):
    _configure_trigger_params(node)
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node, confidence=0.1)) is False


def test_check_trigger_rejects_depth_invalid(node):
    _configure_trigger_params(node)
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node, depth_valid=False)) is False


def test_check_trigger_rejects_not_approaching(node):
    _configure_trigger_params(node)
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node, approaching=False)) is False


def test_check_trigger_rejects_stale_stamp(node):
    _configure_trigger_params(node, **{'trigger.max_track_age_s': 0.5})
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node, age_s=5.0)) is False


def test_check_trigger_rejects_wrong_frame_id(node):
    _configure_trigger_params(node)
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node, frame_id='camera_link')) is False


@pytest.mark.parametrize('x,y,z', [
    (float('nan'), 0.1, 0.1),
    (0.1, float('inf'), 0.1),
    (0.1, 0.1, float('-inf')),
])
def test_check_trigger_rejects_nan_inf_position(node, x, y, z):
    _configure_trigger_params(node)
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node, x=x, y=y, z=z)) is False


def test_check_trigger_rejects_when_min_confidence_unset(node):
    # min_confidence sentinel(-1, 기본값)이면 항상 거부한다(fail-closed).
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node)) is False


def test_check_trigger_rejects_when_required_frame_id_unset(node):
    _configure_trigger_params(node, **{'trigger.required_frame_id': ''})
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node)) is False


def test_check_trigger_rejects_when_max_track_age_unset(node):
    _configure_trigger_params(node, **{'trigger.max_track_age_s': -1.0})
    node.current_tool = 'spanner'

    assert node._check_trigger(_fresh_tool_track(node)) is False


# ---- _get_grasp_spec 실제 구현 ----

def _configure_tool_spec(node, tool, **overrides):
    values = {
        f'tools.{tool}.width_mm': 30.0,
        f'tools.{tool}.force_n': 20.0,
        f'tools.{tool}.verify_min_width_mm': 25.0,
        f'tools.{tool}.verify_max_width_mm': 35.0,
    }
    values.update(overrides)
    node.set_parameters([Parameter(name, value=value) for name, value in values.items()])


def test_get_grasp_spec_returns_configured_values(node):
    _configure_tool_spec(node, 'hammer')

    spec = node._get_grasp_spec('hammer')

    assert spec.width_mm == 30.0
    assert spec.force_n == 20.0
    assert spec.verify_min_width_mm == 25.0
    assert spec.verify_max_width_mm == 35.0


def test_get_grasp_spec_returns_none_when_unconfigured(node):
    # 파라미터 기본값(-1, sentinel)인 상태 - 미설정으로 간주해 None을 반환한다.
    assert node._get_grasp_spec('spanner') is None


def test_get_grasp_spec_returns_none_for_unknown_tool_class(node):
    assert node._get_grasp_spec('unknown_tool') is None


def test_get_grasp_spec_returns_none_when_verify_range_inverted(node):
    _configure_tool_spec(node, 'hammer', **{
        'tools.hammer.verify_min_width_mm': 40.0, 'tools.hammer.verify_max_width_mm': 35.0})

    assert node._get_grasp_spec('hammer') is None


# ---- _verify_grasp 실제 구현 ----

def _make_spec(**overrides):
    defaults = dict(
        width_mm=30.0, force_n=20.0,
        verify_min_width_mm=25.0, verify_max_width_mm=35.0)
    defaults.update(overrides)
    return GraspSpec(**defaults)


def _grasp_result(success=True, grip_detected=True, final_width_mm=30.0):
    result = _FakeResult(success=success)
    result.grip_detected = grip_detected
    result.final_width_mm = final_width_mm
    return result


def test_verify_grasp_succeeds_within_width_range(node):
    node._active_grasp_spec = _make_spec()

    assert node._verify_grasp(_grasp_result(final_width_mm=30.0)) is True


def test_verify_grasp_fails_when_grip_not_detected(node):
    node._active_grasp_spec = _make_spec()

    assert node._verify_grasp(_grasp_result(grip_detected=False)) is False


def test_verify_grasp_fails_when_width_below_range(node):
    node._active_grasp_spec = _make_spec()

    assert node._verify_grasp(_grasp_result(final_width_mm=10.0)) is False


def test_verify_grasp_fails_when_width_above_range(node):
    node._active_grasp_spec = _make_spec()

    assert node._verify_grasp(_grasp_result(final_width_mm=50.0)) is False


def test_verify_grasp_fails_when_spec_missing(node):
    node._active_grasp_spec = None

    assert node._verify_grasp(_grasp_result()) is False


def test_verify_grasp_fails_when_result_not_success(node):
    node._active_grasp_spec = _make_spec()

    assert node._verify_grasp(_grasp_result(success=False)) is False


def test_verify_grasp_fails_on_nonfinite_width(node):
    node._active_grasp_spec = _make_spec()

    assert node._verify_grasp(_grasp_result(final_width_mm=float('nan'))) is False


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


def test_move_safe_result_success_transitions_to_approach_hand(node):
    # handover_safe 도착 후 바로 handover_hold로 가지 않고, 먼저 손에 접근한다
    # (handover_approach) - vision도 TRACK_HAND로 전환한다.
    node.state = State.MOVE_SAFE
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))
    vision_calls = []
    node._set_vision_mode = lambda mode, tool_class='': vision_calls.append(mode)

    node._handle_move_safe_result(_FakeResult(success=True))

    assert node.state == State.APPROACH_HAND
    assert sent == [('handover_approach', {})]
    assert vision_calls == [SetVisionMode.Request.TRACK_HAND]


def test_move_safe_result_failure_sets_fault(node):
    node.state = State.MOVE_SAFE
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_move_safe_result(_FakeResult(success=False, message='motion failed'))

    assert node.safety_state == Safety.FAULT


def test_approach_hand_result_success_transitions_to_wait_pull(node):
    node.state = State.APPROACH_HAND
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_approach_hand_result(_FakeResult(success=True))

    assert node.state == State.WAIT_PULL
    assert sent == [('handover_hold', {})]
    assert node._wait_pull_timeout_timer is not None
    node._wait_pull_timeout_timer.cancel()


def test_approach_hand_result_failure_sets_fault(node):
    node.state = State.APPROACH_HAND
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_approach_hand_result(_FakeResult(success=False, message='lost'))

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


def test_wait_pull_timeout_does_not_cancel_and_stays_in_wait_pull(node):
    # place_down으로 옮기지 않고, handover_hold도 취소하지 않은 채 그 자리에서
    # 계속 들고 대기한다 - GUI 안내만 새로 발행한다.
    node.state = State.WAIT_PULL
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)
    handover_hold_handle = _send_and_accept(node, 'handover_hold')
    node.state = State.WAIT_PULL

    node._on_wait_pull_timeout()

    assert handover_hold_handle.cancel_called is False
    assert node.state == State.WAIT_PULL
    assert node._last_status_detail == WAIT_PULL_REMINDER_MESSAGE
    node._wait_pull_timeout_timer.cancel()


def test_wait_pull_timeout_starts_repeating_reminder_timer(node):
    node.state = State.WAIT_PULL
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)
    node.set_parameters([Parameter('wait_pull_reminder_interval_s', value=10.0)])

    node._on_wait_pull_timeout()

    assert node._wait_pull_timeout_timer is not None
    node._wait_pull_timeout_timer.cancel()


def test_wait_pull_reminder_republishes_message_while_still_waiting(node):
    node.state = State.WAIT_PULL
    node._last_status_detail = ''

    node._on_wait_pull_reminder()

    assert node._last_status_detail == WAIT_PULL_REMINDER_MESSAGE


def test_wait_pull_reminder_stops_once_state_leaves_wait_pull(node):
    node.state = State.HOME
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)

    node._on_wait_pull_reminder()

    assert node._wait_pull_timeout_timer is None  # 더 이상 WAIT_PULL이 아니면 타이머를 정리한다


def test_wait_pull_result_success_still_cancels_reminder_timer(node):
    # pull이 확정되면(성공) 반복 안내 타이머 단계였더라도 정리되고 HOME으로 간다.
    node.state = State.WAIT_PULL
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)
    node._on_wait_pull_timeout()  # 반복 안내 단계로 전환
    node._set_vision_mode = lambda mode, tool_class='': None
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_wait_pull_result(_FakeResult(success=True, message='pull_detected, released'))

    assert node._wait_pull_timeout_timer is None
    assert node.state == State.HOME
    assert sent == [('move_named', {'named_target': 'home'})]


def test_home_result_success_returns_to_idle(node):
    node.state = State.HOME
    node.current_tool = 'spanner'
    node._active_grasp_spec = GraspSpec(
        width_mm=30.0, force_n=20.0,
        verify_min_width_mm=25.0, verify_max_width_mm=35.0)

    node._handle_home_result(_FakeResult(success=True))

    assert node.state == State.IDLE
    assert node.current_tool is None  # 성공 HOME 후 current_tool 정리
    assert node._active_grasp_spec is None


def test_home_result_failure_sets_fault(node):
    node.state = State.HOME
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_home_result(_FakeResult(success=False, message='motion failed'))

    assert node.safety_state == Safety.FAULT


# ---- fetch_tool은 MANUAL 모드에서도 동작 ----

def test_fetch_tool_allowed_in_manual_mode(node):
    node.operation_mode = Mode.MANUAL
    node.set_parameters([Parameter('auto.config_ready', value=True)])
    node._start_after_vision_mode = lambda mode, tool_class, expected_state, on_success: on_success()
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_fetch_tool('spanner')

    assert node.current_tool == 'spanner'
    assert sent == [('move_named', {'named_target': 'watch'})]


# ---- 재개(resume) ----

def test_capture_resume_snapshot_continue_states(node):
    node.current_tool = 'spanner'
    node._active_grasp_spec = _make_spec()
    for state in (State.MOVE_SAFE, State.APPROACH_HAND, State.WAIT_PULL):
        node.state = state
        node._capture_resume_snapshot()
        assert node._resume_kind == 'continue'
        assert node._resume_state == state
        assert node._resume_tool == 'spanner'
        assert node._resume_grasp_spec is not None


def test_capture_resume_snapshot_retry_pick_states(node):
    node.current_tool = 'spanner'
    node._active_grasp_spec = _make_spec()
    for state in (State.SERVO_PICK, State.VERIFY_GRASP):
        node.state = state
        node._capture_resume_snapshot()
        assert node._resume_kind == 'retry_pick'
        assert node._resume_state == state
        assert node._resume_tool == 'spanner'
        assert node._resume_grasp_spec is None


def test_capture_resume_snapshot_none_for_other_states(node):
    node.current_tool = 'spanner'
    node.state = State.DETECT_TRACK
    node._capture_resume_snapshot()

    assert node._resume_kind is None
    assert node._resume_state is None
    assert node._resume_tool is None
    assert node._resume_grasp_spec is None


def test_enter_fault_captures_resume_snapshot_before_state_change(node):
    node.state = State.WAIT_PULL
    node.current_tool = 'wrench'
    node._active_grasp_spec = _make_spec()

    node._enter_fault('FAULT: 예상하지 못한 외력')

    assert node._resume_kind == 'continue'
    assert node._resume_state == State.WAIT_PULL
    assert node._resume_tool == 'wrench'


def test_resumable_field_true_only_when_normal_idle_and_snapshot_present(node):
    node.state = State.WAIT_PULL
    node.current_tool = 'wrench'
    node._active_grasp_spec = _make_spec()
    node._enter_fault('FAULT: 예상하지 못한 외력')

    # FAULT 상태이고 state도 아직 IDLE이 아니다(_enter_fault는 state를 바꾸지 않음).
    published = []
    node.pub_status.publish = published.append
    node._publish_status(detail='')
    assert json.loads(published[-1].data)['resumable'] is False

    # 안전상태만 NORMAL로 돌아왔지만 state가 여전히 WAIT_PULL이면 아직 재개 불가.
    node.safety_state = Safety.NORMAL
    node._publish_status(detail='')
    assert json.loads(published[-1].data)['resumable'] is False

    # state까지 IDLE이 되면 그제서야 재개 가능하다고 보고한다.
    node.state = State.IDLE
    node._publish_status(detail='')
    assert json.loads(published[-1].data)['resumable'] is True


def test_handle_resume_rejected_when_not_normal(node):
    node.safety_state = Safety.FAULT
    node._resume_kind = 'continue'
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_resume()

    assert sent == []
    assert node._resume_kind == 'continue'  # 스냅샷은 그대로 보존된다


def test_handle_resume_rejected_when_not_idle(node):
    node.safety_state = Safety.NORMAL
    node.state = State.MOVE_TO_WATCH
    node._resume_kind = 'continue'
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_resume()

    assert sent == []


def test_handle_resume_no_snapshot_does_nothing(node):
    node.safety_state = Safety.NORMAL
    node.state = State.IDLE
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_resume()

    assert sent == []


def test_handle_resume_continue_wait_pull(node):
    node.safety_state = Safety.NORMAL
    node.state = State.IDLE
    node._resume_kind = 'continue'
    node._resume_state = State.WAIT_PULL
    node._resume_tool = 'wrench'
    node._resume_grasp_spec = _make_spec()
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_resume()

    assert node.state == State.WAIT_PULL
    assert node.current_tool == 'wrench'
    assert node._active_grasp_spec is not None
    assert sent == [('handover_hold', {})]
    assert node._resume_kind is None  # 1회성 - 사용 후 스냅샷을 지운다
    node._wait_pull_timeout_timer.cancel()


def test_handle_resume_continue_move_safe(node):
    node.safety_state = Safety.NORMAL
    node.state = State.IDLE
    node._resume_kind = 'continue'
    node._resume_state = State.MOVE_SAFE
    node._resume_tool = 'spanner'
    node._resume_grasp_spec = _make_spec()
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_resume()

    assert node.state == State.MOVE_SAFE
    assert sent == [('move_named', {'named_target': 'handover_safe'})]


def test_handle_resume_retry_pick_sends_release_and_retry(node):
    node.safety_state = Safety.NORMAL
    node.state = State.IDLE
    node.set_parameters([Parameter('verify_grasp_max_retries', value=2)])
    node._verify_grasp_retries = 0
    node._resume_kind = 'retry_pick'
    node._resume_state = State.SERVO_PICK
    node._resume_tool = 'hammer'
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_resume()

    assert node.state == State.VERIFY_GRASP
    assert node.current_tool == 'hammer'
    assert sent == [('release_and_retry', {})]
    assert node._verify_grasp_retries == 1


def test_handle_resume_retry_pick_exhausted_enters_fault(node):
    node.safety_state = Safety.NORMAL
    node.state = State.IDLE
    node.set_parameters([Parameter('verify_grasp_max_retries', value=0)])
    node._verify_grasp_retries = 0
    node._resume_kind = 'retry_pick'
    node._resume_state = State.VERIFY_GRASP
    node._resume_tool = 'hammer'
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_resume()

    assert sent == []
    assert node.safety_state == Safety.FAULT
    assert node.current_tool is None


def test_home_result_success_clears_stale_resume_snapshot(node):
    node.state = State.HOME
    node._resume_kind = 'continue'
    node._resume_state = State.WAIT_PULL
    node._resume_tool = 'spanner'
    node._resume_grasp_spec = _make_spec()

    node._handle_home_result(_FakeResult(success=True))

    assert node._resume_kind is None
    assert node._resume_state is None
    assert node._resume_tool is None
    assert node._resume_grasp_spec is None


# ---- 일시정지(STOP)가 재개 스냅샷을 남기고, 리셋이 취소+스냅샷 정리를 담당 ----

def test_stop_during_resumable_state_captures_snapshot_and_becomes_resumable(node):
    node.operation_mode = Mode.AUTO
    node.current_tool = 'spanner'
    node._active_grasp_spec = _make_spec()
    node.state = State.MOVE_SAFE
    goal_handle = _send_and_accept(node, 'move_named', named_target='handover_safe')
    node.state = State.MOVE_SAFE

    node._on_user_command(String(data='멈춰'))

    assert node._resume_kind == 'continue'
    assert node.state == State.CANCELLING

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.state == State.IDLE
    published = []
    node.pub_status.publish = published.append
    node._publish_status(detail='')
    assert json.loads(published[-1].data)['resumable'] is True


def test_stop_during_non_resumable_state_leaves_no_snapshot(node):
    node.operation_mode = Mode.MANUAL
    node.state = State.DETECT_TRACK

    node._on_user_command(String(data='멈춰'))

    assert node._resume_kind is None
    assert node.state == State.IDLE


def test_reset_while_normal_and_idle_with_no_snapshot_is_noop(node):
    node.safety_state = Safety.NORMAL
    node.state = State.IDLE
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_reset()

    assert sent == []
    assert node.state == State.IDLE


def test_reset_while_paused_clears_snapshot_and_stays_idle(node):
    # 일시정지로 남은 재개 스냅샷을 리셋이 지운다 - "재개" 없이 완전히 취소.
    node.safety_state = Safety.NORMAL
    node.state = State.IDLE
    node._resume_kind = 'continue'
    node._resume_state = State.WAIT_PULL
    node._resume_tool = 'wrench'
    node._resume_grasp_spec = _make_spec()

    node._handle_reset()

    assert node._resume_kind is None
    assert node._resume_state is None
    assert node._resume_tool is None
    assert node._resume_grasp_spec is None
    assert node.state == State.IDLE


def test_reset_while_task_in_progress_cancels_and_returns_idle_without_snapshot(node):
    # SERVO_PICK은 _capture_resume_snapshot 기준으로는 재개 가능(retry_pick)
    # 분류지만, 리셋은 애초에 스냅샷을 캡처하지 않는다 - 일시정지와 달리 완전
    # 취소가 목적이라 재개할 게 남으면 안 된다(_handle_stop과의 핵심 차이).
    node.operation_mode = Mode.AUTO
    node.state = State.SERVO_PICK
    goal_handle = _send_and_accept(node, 'servo_pick', tool_class='spanner')
    node.state = State.SERVO_PICK

    node._handle_reset()

    assert goal_handle.cancel_called
    assert node.state == State.CANCELLING

    goal_handle.result_future.fire(_FakeResult(success=False, message='canceled'))

    assert node.state == State.IDLE
    assert node._resume_kind is None


# ---- 체크포인트 이벤트 ----

_STATE_CHECKPOINT_CASES = [
    (State.MOVE_TO_WATCH, 'B', 'parse_no_intermediate_state'),
    (State.SERVO_PICK, 'D', 'servo_pick_state_entered'),
    (State.MOVE_SAFE, 'E', 'move_safe_entered'),
    (State.APPROACH_HAND, 'G', 'approach_hand_entered'),
    (State.WAIT_PULL, 'I', 'wait_pull_entered'),
    (State.IDLE, 'K', 'idle_entered'),
]


@pytest.mark.parametrize('new_state,expected_phase,expected_checkpoint', _STATE_CHECKPOINT_CASES)
def test_set_state_publishes_checkpoint_for_known_transitions(
        node, new_state, expected_phase, expected_checkpoint):
    published = []
    node.pub_debug_events.publish = published.append
    node.state = State.MANUAL_MOVE  # IDLE 외의 임의 시작 상태 - 최초 부팅(old_state=None) 가드와 별개로 확인

    node._set_state(new_state, detail='테스트')

    checkpoint_payloads = [
        json.loads(p.data) for p in published if json.loads(p.data)['checkpoint_id'] == expected_checkpoint]
    assert len(checkpoint_payloads) == 1
    assert checkpoint_payloads[0]['phase'] == expected_phase
    assert checkpoint_payloads[0]['status'] == 'PASS'


def test_set_state_guards_against_none_old_state(node):
    """old_state가 None인 채로 _set_state가 불리는 경우(현재 코드에서는
    __init__이 self.state = State.IDLE을 직접 대입하므로 실제로는 일어나지
    않지만, 방어적으로 넣어둔 가드다) K/idle_entered 같은 체크포인트를 잘못
    PASS 처리하지 않아야 한다."""
    published = []
    node.pub_debug_events.publish = published.append
    node.state = None

    node._set_state(State.IDLE, detail='테스트')

    assert published == []


def test_set_state_unmapped_transition_does_not_publish_checkpoint(node):
    published = []
    node.pub_debug_events.publish = published.append
    node.state = State.IDLE

    node._set_state(State.CANCELLING, detail='취소')

    assert published == []


def test_grasp_spec_reject_publishes_servo_pick_triggered_fail(node):
    published = []
    node.pub_debug_events.publish = published.append

    result = node._get_grasp_spec('unsupported_tool_class')

    assert result is None
    payload = json.loads(published[-1].data)
    assert payload['phase'] == 'C'
    assert payload['checkpoint_id'] == 'servo_pick_triggered'
    assert payload['status'] == 'FAIL'


def test_trigger_accept_publishes_servo_pick_triggered_pass(node):
    node.current_tool = 'spanner'
    node.set_parameters([
        Parameter('trigger.min_confidence', value=0.5),
        Parameter('trigger.require_depth_valid', value=False),
        Parameter('trigger.require_approaching', value=False),
        Parameter('trigger.required_frame_id', value='base_link'),
        Parameter('trigger.max_track_age_s', value=10.0),
    ])
    published = []
    node.pub_debug_events.publish = published.append

    msg = ToolTrack()
    msg.tool_class = 'spanner'
    msg.confidence = 0.9
    msg.depth_valid = False
    msg.approaching = False
    msg.header.frame_id = 'base_link'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = 0.1, 0.2, 0.3

    accepted = node._check_trigger(msg)

    assert accepted is True
    payload = json.loads(published[-1].data)
    assert payload['phase'] == 'C'
    assert payload['checkpoint_id'] == 'servo_pick_triggered'
    assert payload['status'] == 'PASS'


def test_servo_pick_result_success_publishes_pass_checkpoint_and_grasp_verified(node):
    node.current_tool = 'spanner'
    node._active_grasp_spec = GraspSpec(
        width_mm=20.0, force_n=10.0, verify_min_width_mm=5.0, verify_max_width_mm=25.0)
    published = []
    node.pub_debug_events.publish = published.append

    result = _FakeResult(success=True)
    result.grip_detected = True
    result.final_width_mm = 15.0
    node._handle_servo_pick_result(result)

    payloads = [json.loads(p.data) for p in published]
    servo_pick = [p for p in payloads if p['checkpoint_id'] == 'servo_pick_result']
    grasp_verified = [p for p in payloads if p['checkpoint_id'] == 'grasp_verified']
    assert servo_pick[0]['phase'] == 'D'
    assert servo_pick[0]['status'] == 'PASS'
    assert grasp_verified[0]['phase'] == 'E'
    assert grasp_verified[0]['status'] == 'PASS'


def test_servo_pick_result_failure_publishes_fail_checkpoint(node):
    published = []
    node.pub_debug_events.publish = published.append

    result = _FakeResult(success=False, message='timeout')
    node._handle_servo_pick_result(result)

    payload = json.loads(published[-1].data)
    assert payload['checkpoint_id'] == 'servo_pick_result'
    assert payload['phase'] == 'D'
    assert payload['status'] == 'FAIL'


def test_verify_grasp_criteria_not_met_publishes_grasp_verified_fail(node):
    node.current_tool = 'spanner'
    node._active_grasp_spec = GraspSpec(
        width_mm=20.0, force_n=10.0, verify_min_width_mm=5.0, verify_max_width_mm=10.0)
    published = []
    node.pub_debug_events.publish = published.append

    result = _FakeResult(success=True)
    result.grip_detected = True
    result.final_width_mm = 50.0  # verify 범위 밖
    node._handle_servo_pick_result(result)

    payloads = [json.loads(p.data) for p in published]
    grasp_verified = [p for p in payloads if p['checkpoint_id'] == 'grasp_verified']
    assert grasp_verified[0]['phase'] == 'E'
    assert grasp_verified[0]['status'] == 'FAIL'


_GOAL_RESULT_CHECKPOINT_CASES = [
    ('move_named', {'named_target': 'watch'}, State.MOVE_TO_WATCH, 'B', 'move_watch_result_received'),
    ('move_named', {'named_target': 'handover_safe'}, State.MOVE_SAFE, 'F', 'handover_safe_result_received'),
    ('handover_approach', {}, State.APPROACH_HAND, 'H', 'handover_approach_result_received'),
    ('handover_hold', {}, State.WAIT_PULL, 'I', 'handover_hold_result_received'),
    ('move_named', {'named_target': 'home'}, State.HOME, 'J', 'home_result_received'),
]


@pytest.mark.parametrize(
    'task_type,kwargs,result_state,expected_phase,expected_checkpoint',
    _GOAL_RESULT_CHECKPOINT_CASES)
def test_goal_result_publishes_checkpoint_on_success(
        node, task_type, kwargs, result_state, expected_phase, expected_checkpoint):
    node.state = result_state
    goal_handle = _send_and_accept(node, task_type, **kwargs)
    published = []
    node.pub_debug_events.publish = published.append

    goal_handle.result_future.fire(_FakeResult(success=True))

    payloads = [json.loads(p.data) for p in published]
    matches = [p for p in payloads if p['checkpoint_id'] == expected_checkpoint]
    assert len(matches) == 1
    assert matches[0]['phase'] == expected_phase
    assert matches[0]['status'] == 'PASS'


def test_goal_result_publishes_fail_checkpoint_on_failure(node):
    node.state = State.MOVE_TO_WATCH
    goal_handle = _send_and_accept(node, 'move_named', named_target='watch')
    published = []
    node.pub_debug_events.publish = published.append

    goal_handle.result_future.fire(_FakeResult(success=False, message='이동 실패'))

    payload = json.loads(published[-1].data)
    assert payload['checkpoint_id'] == 'move_watch_result_received'
    assert payload['status'] == 'FAIL'


def test_goal_result_unmapped_state_does_not_publish_checkpoint(node):
    node.state = State.MANUAL_MOVE
    goal_handle = _send_and_accept(node, 'move_named', named_target='front')
    published = []
    node.pub_debug_events.publish = published.append

    goal_handle.result_future.fire(_FakeResult(success=True))

    payloads = [json.loads(p.data) for p in published]
    assert not any(p['checkpoint_id'].endswith('_result_received') for p in payloads)
