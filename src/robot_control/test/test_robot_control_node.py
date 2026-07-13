import json
import time

import rclpy
import pytest
from rclpy.action import CancelResponse, GoalResponse
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import HandTrack, ToolTrack
from robot_control.hand_servo_loop import HandServoLoop
from robot_control.robot_control_node import (
    DoosanRobotControl, DoosanRobotState, FaultPrefix, RG2Status, RobotControlNode, SafetyState,
)
from robot_control.servo_loop import ServoCommand, ServoLoop


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = RobotControlNode()
    yield n
    n.destroy_node()


class FakeGoalHandle:
    def __init__(self, request):
        self.request = request
        self.succeeded = False
        self.aborted = False
        self.was_canceled = False
        self.is_cancel_requested = False
        self.feedback_msgs = []

    def succeed(self):
        self.succeeded = True

    def abort(self):
        self.aborted = True

    def canceled(self):
        self.was_canceled = True

    def publish_feedback(self, fb):
        self.feedback_msgs.append(fb)


class FakeToolTrackForNode:
    """test_servo_pick_step_consumes_contact_flag_and_locks_z 전용 - ToolTrack의
    header.stamp/pose.position만 필요하므로 test_servo_loop.py의 FakeToolTrack과
    별개로 최소 형태만 둔다(두 테스트 파일 간 임포트 결합을 피하기 위함)."""

    def __init__(self, t, x, y, z):
        self.header = type('H', (), {'stamp': type('S', (), {
            'sec': int(t), 'nanosec': int((t - int(t)) * 1e9)})()})()
        self.pose = type('P', (), {
            'position': type('Pos', (), {'x': x, 'y': y, 'z': z})(),
            'orientation': type('O', (), {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0})()})()
        self.depth_valid = True
        self.yaw_valid = False


def _goal(task_type, named_target=''):
    g = RobotTask.Goal()
    g.task_type = task_type
    g.named_target = named_target
    return g


class _FakeDoosanDriver:
    """dsr_msgs2 없이 RobotControlNode의 오케스트레이션 로직만 검증하기 위한 가짜 드라이버."""

    def __init__(self):
        self.robot_state_sequence = []
        self.set_robot_control_calls = []
        self.ext_torque = [0.0] * 6
        self.tool_force = [0.0] * 6
        self.publish_calls = []
        self.stop_calls = []
        self.stop_return_value = True
        self.stop_should_raise = False

    def get_robot_state(self):
        if self.robot_state_sequence:
            return self.robot_state_sequence.pop(0)
        return DoosanRobotState.STANDBY

    def set_robot_control(self, code):
        self.set_robot_control_calls.append(code)
        return True

    def get_external_torque(self):
        return self.ext_torque

    def get_tool_force(self, ref=0):
        return self.tool_force

    def stop(self, stop_mode=1):
        self.stop_calls.append(stop_mode)
        if self.stop_should_raise:
            raise RuntimeError('move_stop 통신 오류 (fake).')
        return self.stop_return_value

    def publish_speedl(self, cmd, *, accel_param_prefix, period_param_name):
        self.publish_calls.append(cmd)


def _terminal_call_count(gh) -> int:
    """succeed/canceled/abort 중 실제로 호출된 개수를 센다 (정확히 한 번만 호출됐는지 검증용)."""
    return sum([gh.succeeded, gh.was_canceled, gh.aborted])


# ---- move_named ----

def test_move_named_success(node):
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_move_named_failure(node):
    node._call_move_service = lambda **kw: False
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


def test_move_named_unconfigured_named_pose_succeeds_in_dry_run_by_default(node):
    # dry_run.allow_unconfigured_named_poses 기본값(true)에서는 실측 관절값이 없는
    # named pose도 dry-run 상태 흐름 시험을 위해 이동을 허용한다.
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_move_named_unconfigured_named_pose_rejected_when_dry_run_flag_disabled(node):
    node.set_parameters([Parameter('dry_run.allow_unconfigured_named_poses', value=False)])
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


def test_move_named_unconfigured_named_pose_rejected_when_hardware_enabled(node):
    # hardware_enabled=true에서는 dry_run.allow_unconfigured_named_poses와 무관하게
    # 빈 named pose를 절대 허용하지 않는다.
    node.hardware_enabled = True
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


# ---- move_named: 이동 전 그리퍼 open 보장 (home/watch - 물건 가지러 가기 시작점) ----

@pytest.mark.parametrize('named_target', ['home', 'watch'])
def test_move_named_opens_gripper_before_moving_when_not_already_open(node, named_target):
    calls = []
    node._is_gripper_already_open = lambda: False
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True
    node._call_move_service = lambda **kw: calls.append(('move', kw)) or True
    gh = FakeGoalHandle(_goal('move_named', named_target=named_target))

    result = node._execute_move_named(gh)

    assert calls[0] == 'open'
    assert calls[1] == ('move', {'named_target': named_target, 'goal_handle': gh})
    assert gh.succeeded is True
    assert result.success is True


@pytest.mark.parametrize('named_target', ['home', 'watch'])
def test_move_named_skips_open_when_gripper_already_open(node, named_target):
    # 이미 열려있는 상태에서 또 여는 명령을 보내 불필요한 재통신(및 그로 인한
    # 오탐 FAULT 위험)을 만들지 않는다 - _is_gripper_already_open()의 존재 이유.
    calls = []
    node._is_gripper_already_open = lambda: True
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True
    node._call_move_service = lambda **kw: calls.append(('move', kw)) or True
    gh = FakeGoalHandle(_goal('move_named', named_target=named_target))

    result = node._execute_move_named(gh)

    assert calls == [('move', {'named_target': named_target, 'goal_handle': gh})]
    assert gh.succeeded is True
    assert result.success is True


@pytest.mark.parametrize('named_target', ['home', 'watch'])
def test_move_named_cancel_before_opening_gripper_does_not_open(node, named_target):
    calls = []
    node._is_gripper_already_open = lambda: False
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True
    gh = FakeGoalHandle(_goal('move_named', named_target=named_target))
    gh.is_cancel_requested = True

    result = node._execute_move_named(gh)

    assert calls == []
    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False


@pytest.mark.parametrize('named_target', ['home', 'watch'])
def test_move_named_fault_before_opening_gripper_does_not_open(node, named_target):
    calls = []
    node._is_gripper_already_open = lambda: False
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True
    node.safety_state = SafetyState.FAULT
    gh = FakeGoalHandle(_goal('move_named', named_target=named_target))

    result = node._execute_move_named(gh)

    assert calls == []
    assert gh.aborted is True
    assert result.success is False


@pytest.mark.parametrize('named_target', ['home', 'watch'])
def test_move_named_rg2_open_failure_declares_fault(node, named_target):
    node._is_gripper_already_open = lambda: False
    node.rg2_client.open = lambda goal_handle=None: False
    node.rg2_client.last_status = RG2Status.COMMUNICATION_ERROR
    gh = FakeGoalHandle(_goal('move_named', named_target=named_target))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


@pytest.mark.parametrize('named_target', ['home', 'watch'])
def test_move_named_rg2_open_canceled_during_open_ends_as_canceled_not_fault(node, named_target):
    node._is_gripper_already_open = lambda: False
    node.rg2_client.open = lambda goal_handle=None: False
    node.rg2_client.last_status = RG2Status.CANCELED
    gh = FakeGoalHandle(_goal('move_named', named_target=named_target))

    result = node._execute_move_named(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
    assert node.safety_state == SafetyState.NORMAL


def test_move_named_unknown_pose_name_rejected_even_in_dry_run(node):
    gh = FakeGoalHandle(_goal('move_named', named_target='not_a_real_pose'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


# ---- named pose 길이/finite 검사 (값이 채워져 있는 경우) ----

def test_move_named_rejects_five_joint_values(node):
    node._named_poses['watch'] = [0.0, 0.0, 90.0, 0.0, 90.0]  # 6개가 아니라 5개
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


def test_move_named_rejects_seven_joint_values(node):
    node._named_poses['watch'] = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0, 0.0]  # 7개
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), float('-inf')])
def test_move_named_rejects_nan_or_inf_joint_value(node, bad_value):
    node._named_poses['watch'] = [0.0, 0.0, 90.0, bad_value, 90.0, 0.0]
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


def test_move_named_accepts_six_finite_joint_values(node):
    node._named_poses['watch'] = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_move_named_rejects_invalid_joint_values_even_in_dry_run(node):
    # hardware_enabled=false(dry-run)라도 값이 채워져 있다면 길이/finite 검사는
    # 그대로 적용된다 - dry-run 허용은 "값이 아예 비어 있는" 경우에만 적용된다.
    assert node.hardware_enabled is False
    node._named_poses['watch'] = [0.0, 0.0, 90.0]  # 3개뿐 - 채워져 있지만 잘못됨
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


def test_move_named_canceled_calls_canceled_not_abort(node):
    node._call_move_service = lambda **kw: False
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))
    gh.is_cancel_requested = True

    result = node._execute_move_named(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False


def test_move_named_rejected_when_safety_state_not_normal(node):
    node.safety_state = SafetyState.FAULT
    sent = []
    node._call_move_service = lambda **kw: sent.append(kw) or True
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False
    assert sent == []  # 이동을 시도조차 하지 않는다


def test_move_named_real_dry_run_cancel_returns_final_result_promptly(node):
    node._named_poses['watch'] = [0.0] * 6
    node.set_parameters([Parameter('move.dry_run_duration_s', value=1.0)])
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))
    gh.is_cancel_requested = True  # 이동 시작 전부터 취소가 요청된 상태를 시뮬레이션

    result = node._execute_move_named(gh)

    assert gh.was_canceled is True
    assert result.success is False


def test_call_move_service_rejects_success_if_fault_occurred_during_move(node):
    def fake_move_joint(goal_handle, pos, vel, acc):
        # 이동 서비스 자체는 성공을 반환하지만, 그 순간 Fault가 발생했다고 가정한다.
        node.safety_state = SafetyState.FAULT
        return True

    node._named_poses['watch'] = [0.0] * 6
    node._move_joint = fake_move_joint

    success = node._call_move_service(named_target='watch')

    assert success is False  # 이동 함수가 성공을 반환해도 Fault 이후에는 성공 처리하지 않는다


def test_move_named_aborts_not_canceled_when_fault_occurs_during_move(node):
    def fake_move_joint(goal_handle, pos, vel, acc):
        node.safety_state = SafetyState.FAULT
        return True

    node._named_poses['watch'] = [0.0] * 6
    node._move_joint = fake_move_joint
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert gh.was_canceled is False  # cancel이 아니라 Fault이므로 aborted로 구분된다
    assert result.success is False


def test_dry_run_move_returns_false_when_fault_occurs_during_wait(node, monkeypatch):
    import time as time_module
    node.set_parameters([Parameter('move.dry_run_duration_s', value=1.0)])

    def fake_sleep(_s):
        node.safety_state = SafetyState.FAULT

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._dry_run_move(gh)

    assert result is False


def test_release_and_retry_calls_open_and_move_to_watch(node):
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True
    node._call_move_service = lambda **kw: calls.append(('move', kw)) or True
    gh = FakeGoalHandle(_goal('release_and_retry'))

    result = node._execute_release_and_retry(gh)

    assert calls[0] == 'open'
    assert calls[1] == ('move', {'named_target': 'watch', 'goal_handle': gh})
    assert gh.succeeded is True
    assert result.success is True


def test_release_and_retry_cancel_before_open_does_not_open_gripper(node):
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')
    node._call_move_service = lambda **kw: calls.append(('move', kw)) or True
    gh = FakeGoalHandle(_goal('release_and_retry'))
    gh.is_cancel_requested = True

    result = node._execute_release_and_retry(gh)

    assert calls == []  # 취소됐으므로 그리퍼를 열지 않는다
    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False


def test_release_and_retry_rg2_canceled_during_open_ends_as_canceled_not_fault(node):
    # 명령 전 취소 확인과 달리, RG2 open의 busy 대기 "도중"에 취소가 확인된
    # 경우(last_status=CANCELED)에도 FAULT가 아니라 canceled()로 끝나야 한다.
    node.rg2_client.open = lambda goal_handle=None: False
    node.rg2_client.last_status = RG2Status.CANCELED
    gh = FakeGoalHandle(_goal('release_and_retry'))

    result = node._execute_release_and_retry(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
    assert node.safety_state == SafetyState.NORMAL  # FAULT로 처리되지 않았다


def test_release_and_retry_rg2_communication_error_during_open_is_fault(node):
    node.rg2_client.open = lambda goal_handle=None: False
    node.rg2_client.last_status = RG2Status.COMMUNICATION_ERROR
    gh = FakeGoalHandle(_goal('release_and_retry'))

    result = node._execute_release_and_retry(gh)

    assert gh.aborted is True
    assert gh.was_canceled is False
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


def test_release_and_retry_fault_before_open_does_not_open_gripper(node):
    node.safety_state = SafetyState.FAULT
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')
    node._call_move_service = lambda **kw: calls.append(('move', kw)) or True
    gh = FakeGoalHandle(_goal('release_and_retry'))

    result = node._execute_release_and_retry(gh)

    assert calls == []
    assert gh.aborted is True
    assert result.success is False


def test_dispatch_unknown_task_type_aborts(node):
    gh = FakeGoalHandle(_goal('unknown_type'))

    result = node._execute_callback(gh)

    assert gh.aborted is True
    assert result.success is False


def test_dispatch_routes_move_named(node):
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_goal('move_named', named_target='home'))

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_goal_callback_rejects_move_pose_as_unknown_task_type(node):
    # move_pose는 안 쓰는 코드로 정리되어 더 이상 지원하지 않는다 - 알 수 없는
    # task_type으로 거부되어야 한다.
    response = node._goal_callback(_goal('move_pose'))

    assert response == GoalResponse.REJECT


def test_goal_callback_rejects_place_down_as_unknown_task_type(node):
    # place_down은 task_manager가 더 이상 사용하지 않기로 해 정리되었다 - 알 수
    # 없는 task_type으로 거부되어야 한다.
    response = node._goal_callback(_goal('place_down', named_target='place_down'))

    assert response == GoalResponse.REJECT


# ---- goal_callback / cancel_callback (Action 취소 계약) ----

def test_goal_callback_accepts_when_normal_and_idle(node):
    response = node._goal_callback(_goal('move_named', named_target='watch'))

    assert response == GoalResponse.ACCEPT


def test_goal_callback_rejects_when_not_normal_safety_state(node):
    node.safety_state = SafetyState.PROTECTIVE_STOP

    response = node._goal_callback(_goal('move_named', named_target='watch'))

    assert response == GoalResponse.REJECT


def test_goal_callback_rejects_when_goal_already_reserved(node):
    node._goal_reserved = True

    response = node._goal_callback(_goal('move_named', named_target='watch'))

    assert response == GoalResponse.REJECT


def test_goal_callback_rejects_unknown_task_type(node):
    response = node._goal_callback(_goal('not_a_real_task_type'))

    assert response == GoalResponse.REJECT


def test_goal_callback_reserves_atomically_and_releases_after_execute(node):
    first = node._goal_callback(_goal('move_named', named_target='watch'))
    assert first == GoalResponse.ACCEPT
    assert node._goal_reserved is True

    # 아직 execute가 시작되지 않은 상태(예약만 된 상태)에서 두 번째 goal은 거부되어야 한다.
    second = node._goal_callback(_goal('move_named', named_target='watch'))
    assert second == GoalResponse.REJECT

    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))
    node._execute_callback(gh)

    assert node._goal_reserved is False
    third = node._goal_callback(_goal('move_named', named_target='watch'))
    assert third == GoalResponse.ACCEPT


def test_goal_callback_rejects_servo_pick_when_hardware_enabled_and_not_ready(node):
    node.hardware_enabled = True
    # servo_pick.hardware_ready 기본값 False

    response = node._goal_callback(_goal('servo_pick'))

    assert response == GoalResponse.REJECT


def test_cancel_callback_always_accepts(node):
    assert node._cancel_callback(object()) == CancelResponse.ACCEPT


# ---- TCP 위치 캐시 읽기 (_get_current_tcp_posx) ----

def test_get_current_tcp_posx_none_when_hardware_disabled(node):
    # hardware_enabled=false(dry_run)에서는 캐시를 읽지 않고 항상 None을 반환한다.
    assert node.hardware_enabled is False
    node._tcp_pose_cache = {'pos6': [1.0] * 6, 'received_at': time.monotonic()}

    assert node._get_current_tcp_posx() is None


def test_get_current_tcp_posx_none_when_cache_empty(node):
    node.hardware_enabled = True
    node._tcp_pose_cache = None

    assert node._get_current_tcp_posx() is None


def test_get_current_tcp_posx_returns_fresh_cache(node):
    node.hardware_enabled = True
    node._tcp_pose_cache = {
        'pos6': [100.0, 200.0, 300.0, 0.0, 90.0, 0.0],
        'received_at': time.monotonic(),
    }

    assert node._get_current_tcp_posx() == [100.0, 200.0, 300.0, 0.0, 90.0, 0.0]


def test_get_current_tcp_posx_rejects_stale_cache(node, monkeypatch):
    import time as time_module
    clock = {'t': 0.0}
    monkeypatch.setattr(time_module, 'monotonic', lambda: clock['t'])

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.tcp_pose_max_age_s', value=0.1)])
    node._tcp_pose_cache = {'pos6': [100.0, 200.0, 300.0, 0.0, 0.0, 0.0], 'received_at': 0.0}

    clock['t'] = 1.0  # max_age_s(0.1s)를 훌쩍 넘겨 캐시가 오래된 상황을 시뮬레이션

    assert node._get_current_tcp_posx() is None


# ---- servo_pick 서보 명령 계산 (_servo_pick_step) ----

def test_servo_pick_step_passes_full_posx_including_rotation_to_servo_loop_step(node):
    # yaw 제어(ServoLoop.step)가 tcp_pose[5](C각, deg)로 현재 손목 각도를 읽으므로,
    # 위치(x,y,z)만 자르지 않고 회전(A,B,C)까지 그대로 이어붙여 넘겨야 한다 - 예전에는
    # tcp_pose_mm[:3]만 넘겨 yaw_rate가 항상 0으로 나올 수밖에 없었다.
    node.hardware_enabled = True
    node._tcp_pose_cache = {
        'pos6': [100.0, 200.0, 300.0, 1.0, 2.0, 3.0],
        'received_at': time.monotonic(),
    }
    captured = {}
    node.servo_loop.step = lambda tcp_pose, now: captured.setdefault('tcp_pose', tcp_pose)
    node.set_parameters([Parameter('debug.log_servo_decisions', value=False)])

    node._servo_pick_step()

    assert captured['tcp_pose'] == [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]


# ---- TCP 위치 캐시 갱신 (_on_tf_broadcast_timer에 병합됨, 2026-07-08) ----
# 과거에는 _on_tcp_pose_refresh_timer가 별도로 GetCurrentPosx를 폴링했으나, TF
# 방송 폴링과 같은 서비스를 이중 호출해 스레드 고갈을 유발했다(실기 확인) - 이제
# TF 방송용 폴링 결과에 캐시 갱신을 얹으므로 아래 테스트들은 _on_tf_broadcast_timer를
# 통해 캐시 갱신 쪽 동작을 검증한다.

def test_tf_broadcast_timer_does_not_update_cache_when_hardware_disabled(node):
    calls = []

    class _FakeDoosan:
        def get_current_posx(self, ref=0):
            calls.append(True)
            return [1.0] * 6

    node._doosan = _FakeDoosan()  # hardware_enabled은 기본값 False로 유지
    node._tcp_tracking_active = True

    node._on_tf_broadcast_timer()

    assert calls == []
    assert node._tcp_pose_cache is None


def test_tf_broadcast_timer_skips_cache_update_when_servo_pick_not_active(node):
    node.hardware_enabled = True

    class _FakeDoosan:
        def get_current_posx(self, ref=0):
            return [1.0] * 6

    node._doosan = _FakeDoosan()
    node._tcp_tracking_active = False

    node._on_tf_broadcast_timer()

    assert node._tcp_pose_cache is None  # servo_pick이 활성화되지 않으면 캐시는 갱신 안 됨


def test_tf_broadcast_timer_skips_cache_update_when_not_normal(node):
    node.hardware_enabled = True

    class _FakeDoosan:
        def get_current_posx(self, ref=0):
            return [1.0] * 6

    node._doosan = _FakeDoosan()
    node._tcp_tracking_active = True
    node.safety_state = SafetyState.FAULT

    node._on_tf_broadcast_timer()

    assert node._tcp_pose_cache is None  # Fault 중에는 캐시를 갱신하지 않는다


def test_tf_broadcast_timer_updates_cache_on_success(node):
    node.hardware_enabled = True

    class _FakeDoosan:
        def get_current_posx(self, ref=0):
            return [10.0, 20.0, 30.0, 0.0, 0.0, 0.0]

    node._doosan = _FakeDoosan()
    node._tcp_tracking_active = True

    node._on_tf_broadcast_timer()

    assert node._tcp_pose_cache['pos6'] == [10.0, 20.0, 30.0, 0.0, 0.0, 0.0]
    assert node._get_current_tcp_posx() == [10.0, 20.0, 30.0, 0.0, 0.0, 0.0]


def test_tf_broadcast_timer_does_not_reuse_stale_value_after_failed_lookup(node):
    # 이전에 성공한 캐시가 있는 상태에서 이후 조회가 계속 실패하면, 캐시를 그대로
    # 두어(덮어쓰지 않아) 나이가 계속 늘어나고 결국 _get_current_tcp_posx가 거부하게
    # 한다 - 실패를 성공한 것처럼 새로 반영하지 않는다.
    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.tcp_pose_max_age_s', value=0.0)])
    node._tcp_tracking_active = True

    class _FailingDoosan:
        def get_current_posx(self, ref=0):
            return None  # 서비스 미준비/timeout/success=false 등

    node._doosan = _FailingDoosan()
    node._tcp_pose_cache = {'pos6': [1.0] * 6, 'received_at': time.monotonic()}

    node._on_tf_broadcast_timer()

    # max_age_s=0.0이므로 이미 존재하던 캐시도 신선하지 않다고 거부되어야 한다
    # (실패한 조회가 캐시를 새로 신선하게 만들지 않았음을 함께 증명한다).
    assert node._get_current_tcp_posx() is None


def test_tf_broadcast_timer_serializes_overlapping_requests(node):
    # servo_pick 실행 중 타이머가 재진입해도 이미 요청이 진행 중이면 새
    # GetCurrentPosx 호출을 겹쳐서 시작하지 않는다.
    calls = []

    class _FakeDoosan:
        def get_current_posx(self, ref=0):
            calls.append(True)
            # 요청이 아직 "진행 중"인 것처럼 재진입을 시도한다.
            node._on_tf_broadcast_timer()
            return [1.0] * 6

    node.hardware_enabled = True
    node._doosan = _FakeDoosan()
    node._tcp_tracking_active = True

    node._on_tf_broadcast_timer()

    assert len(calls) == 1  # 재진입 시도는 in_flight 플래그로 걸러져 겹치지 않는다


def test_servo_pick_execution_sets_and_clears_tcp_tracking_active(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._servo_pick_tick = lambda: ('CLOSE', None)
    node.servo_loop.get_state = lambda: 'closing'
    node.servo_loop.start = lambda *a, **k: None
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)

    observed_during_execution = {'active': None}

    def fake_tick():
        observed_during_execution['active'] = node._tcp_tracking_active
        return ('CLOSE', None)

    node._servo_pick_tick = fake_tick

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    assert node._tcp_tracking_active is False
    node._execute_servo_pick(gh)

    assert observed_during_execution['active'] is True  # 실행 중에는 True였다
    assert node._tcp_tracking_active is False  # 종료 후에는 반드시 해제된다


# ---- ToolTrack 유효성 검사 (_validate_tool_track_message) ----
# 칼만 ServoLoop는 msg.pose.position을 base_link 기준 절대 목표 위치로 직접 필터에
# 흘려보낸다(TCP 오차로 변환하지 않음) - 여기서는 frame_id/NaN만 확인한다.

def _tool_track(frame_id='base_link', x=1.5, y=0.2, z=0.4):
    msg = ToolTrack()
    msg.header.frame_id = frame_id
    msg.tool_class = 'spanner'
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    msg.depth_valid = True
    msg.approaching = True
    msg.confidence = 0.9
    return msg


def test_validate_tool_track_message_accepts_valid_message(node):
    assert node._validate_tool_track_message(_tool_track(x=1.5, y=0.2, z=0.4)) is True


def test_validate_tool_track_message_rejects_wrong_frame_id(node):
    assert node._validate_tool_track_message(_tool_track(frame_id='camera_link')) is False


@pytest.mark.parametrize('x,y,z', [
    (float('nan'), 0.2, 0.4),
    (1.5, float('inf'), 0.4),
    (1.5, 0.2, float('-inf')),
])
def test_validate_tool_track_message_rejects_nan_inf_position(node, x, y, z):
    assert node._validate_tool_track_message(_tool_track(x=x, y=y, z=z)) is False


def test_on_tool_track_during_servo_ignores_invalid_message(node):
    node._validate_tool_track_message = lambda msg: False
    received = []
    node.servo_loop.on_tool_track = lambda msg: received.append(msg)

    node._on_tool_track_during_servo(_tool_track())

    assert received == []


def test_on_tool_track_during_servo_forwards_raw_message_to_servo_loop(node):
    msg = _tool_track()
    received = []
    node.servo_loop.on_tool_track = lambda m: received.append(m)

    node._on_tool_track_during_servo(msg)

    assert received == [msg]


# ---- handover_approach goal 거부 (hardware_ready 게이트) ----

def test_goal_callback_rejects_handover_approach_when_hardware_ready_false(node):
    node.hardware_enabled = True

    response = node._goal_callback(_goal('handover_approach'))

    assert response == GoalResponse.REJECT


def test_goal_callback_accepts_handover_approach_in_dry_run(node):
    assert node.hardware_enabled is False

    response = node._goal_callback(_goal('handover_approach'))

    assert response == GoalResponse.ACCEPT


# ---- handover_approach 실행 (성공/취소/예외) ----

def test_execute_handover_approach_rejected_when_hardware_ready_false(node):
    node.hardware_enabled = True
    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.aborted is True
    assert result.success is False


def test_execute_handover_approach_rejected_when_safety_state_not_normal(node):
    node.safety_state = SafetyState.FAULT
    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.aborted is True
    assert result.success is False


def test_execute_handover_approach_dry_run_succeeds(node):
    # hardware_enabled=False(기본) - hand_track 없이도 흐름만 검증하고 바로 도착 처리.
    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_execute_handover_approach_dry_run_canceled(node, monkeypatch):
    import time as time_module
    node.set_parameters([Parameter('move.dry_run_duration_s', value=1.0)])
    gh = FakeGoalHandle(_goal('handover_approach'))

    def fake_sleep(_s):
        gh.is_cancel_requested = True

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)

    result = node._execute_handover_approach(gh)

    assert gh.was_canceled is True
    assert result.success is False


# ---- HandTrack 유효성 검사 (_validate_hand_track_message) ----
# HandServoLoop는 msg.pose.position을 base_link 기준 절대 손 위치로 직접 사용한다
# (TCP 오차로 변환하지 않음) - 여기서는 frame_id/NaN만 확인한다.

def _hand_track(frame_id='base_link', x=1.0, y=0.2, z=0.4, detected=True, fist=False):
    msg = HandTrack()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    msg.detected = detected
    msg.fist = fist
    msg.confidence = 0.9
    return msg


def test_validate_hand_track_message_accepts_valid_message(node):
    assert node._validate_hand_track_message(_hand_track(x=1.5, y=0.2, z=0.4)) is True


def test_validate_hand_track_message_rejects_wrong_frame_id(node):
    assert node._validate_hand_track_message(_hand_track(frame_id='camera_link')) is False


@pytest.mark.parametrize('x,y,z', [
    (float('nan'), 0.2, 0.4),
    (1.5, float('inf'), 0.4),
    (1.5, 0.2, float('-inf')),
])
def test_validate_hand_track_message_rejects_nan_inf_position(node, x, y, z):
    assert node._validate_hand_track_message(_hand_track(x=x, y=y, z=z)) is False


def test_on_hand_track_during_servo_ignores_invalid_message(node):
    node._validate_hand_track_message = lambda msg: False
    received = []
    node.hand_servo_loop.on_hand_track = lambda msg: received.append(msg)

    node._on_hand_track_during_servo(_hand_track())

    assert received == []


def test_on_hand_track_during_servo_forwards_raw_message_to_hand_servo_loop(node):
    msg = _hand_track()
    received = []
    node.hand_servo_loop.on_hand_track = lambda m: received.append(m)

    node._on_hand_track_during_servo(msg)

    assert received == [msg]


# ---- handover_approach tick (_handover_approach_tick) ----

def test_handover_approach_tick_continue(node):
    node.hand_servo_loop.tick = lambda: ('CONTINUE', None)

    status, reason = node._handover_approach_tick()

    assert status == 'CONTINUE'
    assert reason is None


def test_handover_approach_tick_stop_on_fist(node):
    node.hand_servo_loop.tick = lambda: ('STOP', 'fist_detected')

    status, reason = node._handover_approach_tick()

    assert status == 'STOP'
    assert reason == 'fist_detected'


def test_handover_approach_tick_abort(node):
    node.hand_servo_loop.tick = lambda: ('ABORT', 'hand_lost')

    status, reason = node._handover_approach_tick()

    assert status == 'ABORT'
    assert reason == 'hand_lost'


# ---- handover_servo 속도 명령 검증 (_validate_handover_servo_command) ----

def test_validate_handover_servo_command_rejects_nan_inf(node):
    assert node._validate_handover_servo_command(ServoCommand(vx=float('nan'))) is False
    assert node._validate_handover_servo_command(ServoCommand(vz=float('inf'))) is False


def test_validate_handover_servo_command_rejects_over_v_max(node):
    v_max = node.get_parameter('handover_servo.v_max').value
    assert node._validate_handover_servo_command(ServoCommand(vx=v_max * 10)) is False
    assert node._validate_handover_servo_command(ServoCommand(vz=v_max * 10)) is False


def test_validate_handover_servo_command_accepts_within_limits(node):
    v_max = node.get_parameter('handover_servo.v_max').value
    cmd = ServoCommand(vx=v_max * 0.5, vy=-v_max * 0.5, vz=v_max * 0.3, yaw_rate=0.0)

    assert node._validate_handover_servo_command(cmd) is True


def _enable_handover_servo_hardware(node):
    node.hardware_enabled = True
    node.set_parameters([Parameter('handover_servo.hardware_ready', value=True)])


def test_handover_approach_execution_sets_and_clears_tcp_tracking_active(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _enable_handover_servo_hardware(node)

    observed_during_execution = {'active': None}

    def fake_tick():
        observed_during_execution['active'] = node._tcp_tracking_active
        return ('STOP', None)

    node._handover_approach_tick = fake_tick
    node.hand_servo_loop.get_state = lambda: 'stopping'
    node.hand_servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('handover_approach'))

    assert node._tcp_tracking_active is False
    node._execute_handover_approach(gh)

    assert observed_during_execution['active'] is True  # 실행 중에는 True였다
    assert node._tcp_tracking_active is False  # 종료 후에는 반드시 해제된다


def test_execute_handover_approach_stops_on_fist_and_returns_result(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _enable_handover_servo_hardware(node)

    ticks = iter(['CONTINUE', 'CONTINUE', 'STOP'])
    node._handover_approach_tick = lambda: (next(ticks), None)
    node.hand_servo_loop.step = lambda *a, **k: None
    node.hand_servo_loop.get_state = lambda: 'following'
    node.hand_servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.succeeded is True
    assert result.success is True
    assert len(gh.feedback_msgs) == 3


def test_execute_handover_approach_calls_move_stop_after_fist_stop(node, monkeypatch):
    # 주먹 확정(STOP)은 긴급정지가 아니라 의도된 정지지만, HandServoLoop의 STOP은
    # servo_pick의 CLOSE와 달리 속도 수렴(should_close 같은 조건)을 요구하지 않으므로
    # _run_rt_tracking이 break만 하고 끝내면 로봇이 마지막 속도로 계속 움직일 수 있다 -
    # 그래서 ARRIVED 직후 명시적으로 MoveStop을 호출해야 한다.
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _enable_handover_servo_hardware(node)
    fake = _FakeDoosanDriver()
    node._doosan = fake
    node._handover_approach_tick = lambda: ('STOP', None)
    node.hand_servo_loop.get_state = lambda: 'stopping'
    node.hand_servo_loop.start = lambda *a, **k: None
    node.hand_servo_loop.step = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.succeeded is True
    assert result.success is True
    assert fake.stop_calls == [node.get_parameter('safety.fault_stop_mode').value]


def test_execute_handover_approach_declares_fault_when_stop_motion_fails_after_fist(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _enable_handover_servo_hardware(node)
    fake = _FakeDoosanDriver()
    fake.stop_return_value = False
    node._doosan = fake
    node._handover_approach_tick = lambda: ('STOP', None)
    node.hand_servo_loop.get_state = lambda: 'stopping'
    node.hand_servo_loop.start = lambda *a, **k: None
    node.hand_servo_loop.step = lambda *a, **k: None
    published = []
    node.pub_fault.publish = published.append

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.aborted is True
    assert gh.succeeded is False
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT
    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.FAULT)


def test_execute_handover_approach_abort_returns_reason(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _enable_handover_servo_hardware(node)

    node._handover_approach_tick = lambda: ('ABORT', 'hand_lost')
    node.hand_servo_loop.get_state = lambda: 'approaching'
    node.hand_servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'hand_lost'


def test_execute_handover_approach_cancel_mid_loop_calls_canceled(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _enable_handover_servo_hardware(node)

    node.hand_servo_loop.get_state = lambda: 'approaching'
    node.hand_servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('handover_approach'))
    gh.is_cancel_requested = True

    result = node._execute_handover_approach(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False


def test_handover_approach_aborts_on_hand_lost(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _enable_handover_servo_hardware(node)

    node.hand_servo_loop = HandServoLoop(
        kp_xy=1.2, kp_z=1.2, v_max=0.15, offset_m=0.2, t_lost_s=0.0, timeout_s=5.0)
    # HandServoLoop.tick()은 on_hand_track이 한 번도 불리지 않으면 _last_msg_time이
    # None으로 남아 'hand_lost'가 판정되지 않는다 - start() 직후 메시지를 하나
    # 흘려보내 유실 타이머가 실제로 돌게 만든다.
    real_start = node.hand_servo_loop.start

    def _start_and_seed(*args, **kwargs):
        real_start(*args, **kwargs)
        node.hand_servo_loop.on_hand_track(_hand_track())

    node.hand_servo_loop.start = _start_and_seed

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'hand_lost'


def test_dispatch_routes_handover_approach(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _enable_handover_servo_hardware(node)

    node._handover_approach_tick = lambda: ('STOP', None)
    node.hand_servo_loop.get_state = lambda: 'stopping'
    node.hand_servo_loop.start = lambda *a, **k: None
    node.hand_servo_loop.step = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


# ---- SpeedlStream 발행 직전 마지막 안전 검사 (_validate_servo_command) ----

def test_validate_servo_command_rejects_nan_inf(node):
    assert node._validate_servo_command(ServoCommand(vx=float('nan'))) is False
    assert node._validate_servo_command(ServoCommand(vy=float('inf'))) is False


def test_validate_servo_command_rejects_over_v_max(node):
    v_max = node.get_parameter('servo.v_max').value
    assert node._validate_servo_command(ServoCommand(vx=v_max * 10)) is False


def test_validate_servo_command_rejects_over_descend_speed(node):
    descend_speed = node.get_parameter('servo.descend_speed').value
    assert node._validate_servo_command(ServoCommand(vz=-descend_speed * 10)) is False


def test_validate_servo_command_accepts_within_limits(node):
    v_max = node.get_parameter('servo.v_max').value
    descend_speed = node.get_parameter('servo.descend_speed').value
    yaw_rate_max = node.get_parameter('servo.yaw_rate_max_deg_s').value
    cmd = ServoCommand(
        vx=v_max * 0.5, vy=-v_max * 0.5, vz=-descend_speed, yaw_rate=yaw_rate_max * 0.5)

    assert node._validate_servo_command(cmd) is True


def test_validate_servo_command_rejects_over_yaw_rate_max(node):
    # yaw_rate(deg/s)는 v_max(m/s)와 단위가 달라 별도 한도로 검증돼야 한다 - 예전에는
    # v_max를 그대로 재사용해 이 케이스가 걸러지지 않았다(yaw_rate가 항상 0이라 드러나지
    # 않던 버그, servo_loop.py의 kp_yaw 활성화와 함께 수정됨).
    yaw_rate_max = node.get_parameter('servo.yaw_rate_max_deg_s').value
    assert node._validate_servo_command(ServoCommand(yaw_rate=yaw_rate_max * 10)) is False


def test_servo_loop_wires_innovation_and_kalman_params_from_ros_params(node):
    assert node.servo_loop.innov_low == node.get_parameter('servo.innov_low').value
    assert node.servo_loop.innov_high == node.get_parameter('servo.innov_high').value
    assert node.servo_loop.w_alpha == node.get_parameter('servo.w_alpha').value
    assert node.servo_loop.z_close == node.get_parameter('servo.z_close').value
    assert node.servo_loop.diverge_n == node.get_parameter('servo.diverge_n').value
    assert (node.servo_loop.diverge_min_delta_m
            == node.get_parameter('servo.diverge_min_delta_m').value)
    assert node.servo_loop.descend_accel_m_s2 == pytest.approx(
        node.get_parameter('servo_pick.speedl_acc_trans_mm_s2').value / 1000.0)
    assert (node.servo_loop.descend_stop_margin_m
            == node.get_parameter('servo.descend_stop_margin_m').value)
    assert node.servo_loop.cov_threshold == node.get_parameter('servo.cov_threshold').value
    assert node.servo_loop._filter.q_pos == node.get_parameter('servo.kalman_q_pos').value
    assert node.servo_loop._filter.q_vel == node.get_parameter('servo.kalman_q_vel').value
    assert node.servo_loop._filter.r_xy == node.get_parameter('servo.kalman_r_xy').value
    assert node.servo_loop._filter.r_z == node.get_parameter('servo.kalman_r_z').value
    assert (node.servo_loop._filter.p0_vel_reset
            == node.get_parameter('servo.kalman_p0_vel_reset').value)
    assert (node.servo_loop.yaw_rate_max_deg_s
            == node.get_parameter('servo.yaw_rate_max_deg_s').value)
    assert node.servo_loop.eps_yaw_deg == node.get_parameter('servo.eps_yaw_deg').value
    assert node.servo_loop.n_stable_yaw == node.get_parameter('servo.n_stable_yaw').value
    assert node.servo_loop.yaw_sign == node.get_parameter('servo.yaw_sign').value
    assert node.servo_loop.yaw_offset_deg == node.get_parameter('servo.yaw_offset_deg').value


# ---- servo_pick ----

def test_servo_pick_tick_continue(node):
    node.servo_loop.should_abort = lambda: None
    node.servo_loop.should_close = lambda: False

    status, reason = node._servo_pick_tick()

    assert status == 'CONTINUE'
    assert reason is None


def test_servo_pick_tick_close(node):
    node.servo_loop.should_abort = lambda: None
    node.servo_loop.should_close = lambda: True

    status, reason = node._servo_pick_tick()

    assert status == 'CLOSE'


def test_servo_pick_tick_abort(node):
    node.servo_loop.should_abort = lambda: 'diverged'

    status, reason = node._servo_pick_tick()

    assert status == 'ABORT'
    assert reason == 'diverged'


def test_grasp_lock_keeps_tracking_while_gripper_closes_then_closes_once_done(node):
    """RG2 close()가 그리퍼가 실제로 멈출 때까지 블로킹하는 동안에도, 그 완료를
    기다리는 매 틱마다 'CONTINUE'를 반환해 x,y 시각 서보 추적(step 호출)이 멈추지
    않아야 하고, 그리퍼가 완전히 닫힌 뒤에야 'CLOSE'를 반환해야 한다."""
    import threading

    close_started = threading.Event()
    close_release = threading.Event()

    def _fake_close(width, force, goal_handle=None):
        close_started.set()
        assert close_release.wait(timeout=2.0), 'close_release가 제때 set되지 않았습니다'
        return True

    node.rg2_client.close = _fake_close
    node.servo_loop.should_abort = lambda: None
    node.servo_loop.should_close = lambda: True

    gh = FakeGoalHandle(_goal('servo_pick'))
    request = gh.request
    request.grasp_width_mm = 30.0
    request.grasp_force_n = 20.0
    node._servo_pick_close_thread = None
    node._servo_pick_close_success = None

    status, reason = node._servo_pick_tick_with_grasp_lock(gh, request)
    assert status == 'CONTINUE'  # close 스레드를 시작만 하고 루프는 계속돼야 한다
    assert close_started.wait(timeout=2.0) is True

    # 그리퍼가 여전히 닫히는 중(close_release 대기 중)이면 계속 추적을 유지해야 한다.
    for _ in range(3):
        status, reason = node._servo_pick_tick_with_grasp_lock(gh, request)
        assert status == 'CONTINUE'

    close_release.set()
    node._servo_pick_close_thread.join(timeout=2.0)

    status, reason = node._servo_pick_tick_with_grasp_lock(gh, request)
    assert status == 'CLOSE'  # 그리퍼 폐합이 끝난 뒤에야 종료 신호를 보낸다
    assert node._servo_pick_close_success is True


def test_grasp_lock_defers_close_when_cancel_requested_right_at_close(node):
    gh = FakeGoalHandle(_goal('servo_pick'))
    request = gh.request
    request.grasp_width_mm = 30.0
    request.grasp_force_n = 20.0
    node._servo_pick_close_thread = None
    node._servo_pick_close_success = None

    node.servo_loop.should_abort = lambda: None
    node.servo_loop.should_close = lambda: True
    closed = []
    node.rg2_client.close = lambda width, force, goal_handle=None: closed.append(True)
    gh.is_cancel_requested = True

    status, reason = node._servo_pick_tick_with_grasp_lock(gh, request)

    assert status == 'CONTINUE'  # 취소 요청이 있으면 그리퍼를 닫기 시작하지 않는다
    assert closed == []
    assert node._servo_pick_close_thread is None


def test_execute_servo_pick_success_closes_gripper_and_returns_result(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    ticks = iter(['CONTINUE', 'CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node.servo_loop.step = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    node._get_current_tcp_pose = lambda: (0.0, 0.0, 0.05, 0, 0, 0)
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (29.4, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.succeeded is True
    assert result.success is True
    assert result.final_width_mm == 29.4
    assert result.grip_detected is True
    # should_close 감지 이후 RG2 close 완료를 확인하기까지 최소 한 틱이 더 필요하므로
    # (그 사이에도 x,y 추적을 계속하기 위함) 정확히 3회가 아니라 최소 3회 이상이다.
    assert len(gh.feedback_msgs) >= 3


def test_servo_pick_rg2_canceled_during_close_ends_as_canceled_not_fault(node, monkeypatch):
    # RG2 close의 busy 대기 "도중"에 취소가 확인된 경우(last_status=CANCELED)에도
    # FAULT가 아니라 기존 cleanup 경로(_finish_cancel)를 통해 canceled()로 끝나야 한다.
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    ticks = iter(['CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node.servo_loop.step = lambda: None
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    def _fake_close(width, force, goal_handle=None):
        node.rg2_client.last_status = RG2Status.CANCELED
        return False

    node.rg2_client.close = _fake_close

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
    assert node.safety_state == SafetyState.NORMAL  # FAULT로 처리되지 않았다


def test_execute_servo_pick_abort_returns_reason(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._servo_pick_tick = lambda: ('ABORT', 'diverged')
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'diverged'


def test_execute_servo_pick_abort_calls_move_stop_when_hardware_enabled(node, monkeypatch):
    # diverging/timeout/tracking_lost 같은 일반 ABORT도 취소 경로와 동일하게 실제
    # MoveStop을 걸어야 한다 - 안 그러면 로봇이 마지막 속도로 계속 움직인다
    # (2026-07-10 실기: diverging abort 후 관성 하강으로 바닥 충돌 확인).
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    node._servo_pick_tick = lambda: ('ABORT', 'diverged')
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'diverged'
    assert fake.stop_calls == [node.get_parameter('safety.recoverable_stop_mode').value]


def test_execute_servo_pick_abort_declares_fault_when_move_stop_fails(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    fake.stop_return_value = False
    node._doosan = fake
    node._servo_pick_tick = lambda: ('ABORT', 'diverged')
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    published = []
    node.pub_fault.publish = published.append

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT
    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.FAULT)


def test_execute_servo_pick_cancel_mid_loop_calls_canceled(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False


def test_execute_servo_pick_cancel_calls_real_move_stop_when_hardware_enabled(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is True
    assert result.success is False
    assert fake.stop_calls == [node.get_parameter('safety.recoverable_stop_mode').value]


# ---- servo_pick: cleanup(MoveStop/subscription) 실패 시 취소 성공으로 가장하지 않음 ----

def test_servo_pick_cancel_aborts_as_fault_when_move_stop_returns_false(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    fake.stop_return_value = False
    node._doosan = fake
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    published = []
    node.pub_fault.publish = published.append

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is False  # 취소 성공으로 가장하지 않는다
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert result is not None
    assert node.safety_state == SafetyState.FAULT
    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.FAULT)


def test_servo_pick_cancel_aborts_as_fault_when_move_stop_raises(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    fake.stop_should_raise = True
    node._doosan = fake
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is False
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


def test_cleanup_destroy_subscription_catches_exception(node):
    def _raise_destroy(sub):
        raise RuntimeError('destroy_subscription boom')

    node.destroy_subscription = _raise_destroy

    assert node._cleanup_destroy_subscription(object()) is False
    assert node._cleanup_destroy_subscription(None) is True  # None은 정리할 것이 없어 성공 취급


def test_servo_pick_cancel_aborts_as_fault_when_subscription_removal_fails(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    # 실제 destroy_subscription을 건드리지 않고(테스트 종료 시 node.destroy_node()가
    # 남은 구독을 정리하려다 다시 실패하는 것을 피하기 위해) 정리 경계 wrapper만
    # 실패하도록 대체한다.
    node._cleanup_destroy_subscription = lambda sub: False
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is False
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


def test_execute_servo_pick_cancel_right_before_closing_gripper_does_not_close(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    def tick():
        # CLOSE 판정 직후, 그리퍼를 실제로 닫기 전에 취소가 들어온 상황을 시뮬레이션한다.
        gh.is_cancel_requested = True
        return ('CLOSE', None)

    node._servo_pick_tick = tick
    node.servo_loop.step = lambda: None
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    closed = []
    node.rg2_client.close = lambda width, force, goal_handle=None: closed.append((width, force))

    result = node._execute_servo_pick(gh)

    assert closed == []  # 그리퍼를 닫지 않는다
    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False


def test_execute_servo_pick_fault_right_before_closing_gripper_does_not_close(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    def tick():
        node.safety_state = SafetyState.FAULT
        return ('CLOSE', None)

    node._servo_pick_tick = tick
    node.servo_loop.step = lambda: None
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    closed = []
    node.rg2_client.close = lambda width, force, goal_handle=None: closed.append((width, force))

    result = node._execute_servo_pick(gh)

    assert closed == []
    assert gh.aborted is True
    assert gh.was_canceled is False
    assert result.success is False


def test_servo_pick_aborts_on_tracking_loss(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.servo_loop = ServoLoop(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.1,
                                 eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                                 dt_latency=0.05, timeout_s=5.0, t_lost_s=0.0)
    # 칼만 ServoLoop는 on_tool_track이 한 번도 불리지 않으면 _last_msg_time이
    # None으로 남아 'tracking_lost'가 판정되지 않는다 - start() 직후 메시지를
    # 하나 흘려보내 추적 유실 타이머가 실제로 돌게 만든다.
    real_start = node.servo_loop.start

    def _start_and_seed(*args, **kwargs):
        real_start(*args, **kwargs)
        node.servo_loop.on_tool_track(_tool_track())

    node.servo_loop.start = _start_and_seed

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'tracking_lost'


def test_servo_pick_rejected_when_safety_state_not_normal(node):
    node.safety_state = SafetyState.FAULT
    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False


def test_dispatch_routes_servo_pick(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._servo_pick_tick = lambda: ('CLOSE', None)
    node.servo_loop.get_state = lambda: 'closing'
    node.servo_loop.start = lambda *a, **k: None
    node.servo_loop.step = lambda *a, **k: None
    node._get_current_tcp_pose = lambda: (0.0, 0.0, 0.05, 0, 0, 0)
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


# ---- servo_pick: hardware_ready 게이트 / RT 세션 확인 ----

def test_servo_pick_rejected_when_hardware_enabled_but_not_ready(node):
    node.hardware_enabled = True
    # servo_pick.hardware_ready 기본값 False

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert 'hardware_ready' in result.message


def test_servo_pick_dry_run_still_works_when_hardware_disabled(node, monkeypatch):
    # hardware_enabled=false(기본값)이면 hardware_ready 게이트와 무관하게 dry-run이 동작해야 한다.
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._servo_pick_tick = lambda: ('CLOSE', None)
    node.servo_loop.start = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'closing'
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_servo_pick_publishes_speedl_only_when_hardware_ready(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([
        Parameter('servo_pick.hardware_ready', value=True),
        # 이 테스트는 그리퍼 폐합 전 x,y 추적 단계의 speedl 발행 게이팅만 검증하는
        # 것이라, 새로 추가된 들어올림 단계(_run_grasp_lift)는 꺼둔다 - 안 그러면
        # 아래 고정된 fake TCP z값이 절대 움직이지 않아 들어올림이 목표 높이에
        # 도달하지 못해 타임아웃(FAULT)으로 끝난다.
        Parameter('servo_pick.lift_height_m', value=0.0),
    ])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    ticks = iter(['CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node._get_current_tcp_posx = lambda: [0.0, 0.0, 50.0, 0.0, 0.0, 0.0]  # mm
    node.servo_loop.step = lambda *a, **k: ServoCommand(vx=0.1)
    node.servo_loop.start = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'closing'
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert result.success is True
    # should_close 감지(CLOSE) 이후에도 RG2 close 완료가 확인될 때까지 x,y 추적이
    # 계속되며 speedl을 계속 발행하므로(그 사이 최소 한 번 더 publish), 정확히
    # 1회가 아니라 최소 1회 이상이다.
    assert len(fake.publish_calls) >= 1


def test_servo_pick_lift_after_grip_moves_up_before_returning_result(node, monkeypatch):
    """그리퍼 폐합 확인 후 VERIFY_GRASP 판정 전에 z를 lift_height_m만큼 들어올려야
    한다(docs/전체 계획.md "즉시 들어올림", 2026-07-12 결정) - TCP z가 목표
    높이에 도달할 때까지 양의 vz speedl을 발행하고, 도달하면 결과를 성공으로
    반환해야 한다."""
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([
        Parameter('servo_pick.hardware_ready', value=True),
        Parameter('servo_pick.lift_height_m', value=0.05),
    ])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    ticks = iter(['CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node.servo_loop.step = lambda *a, **k: None
    node.servo_loop.start = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'closing'
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)

    # 50.0mm에서 시작해 매 호출마다 10mm씩 상승 - lift_height_m(0.05m=50mm)만큼
    # 오른 100.0mm에서 목표 도달.
    z_values = iter(50.0 + 10.0 * i for i in range(20))
    node._get_current_tcp_posx = lambda: [0.0, 0.0, next(z_values), 0.0, 0.0, 0.0]

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert result.success is True
    lift_calls = [c for c in fake.publish_calls if c.vz > 0.0]
    assert len(lift_calls) >= 1
    assert any(fb.state == 'lifting' for fb in gh.feedback_msgs)


def test_servo_pick_lift_times_out_as_fault_when_z_never_reaches_target(node, monkeypatch):
    """TCP z 피드백이 멈춰 목표 높이에 영영 도달하지 못하면(하드웨어 이상 등),
    무한 루프에 빠지지 않고 lift_timeout_s 안에 FAULT로 끝나야 한다."""
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([
        Parameter('servo_pick.hardware_ready', value=True),
        Parameter('servo_pick.lift_height_m', value=0.05),
        Parameter('servo_pick.lift_timeout_s', value=0.05),
    ])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    ticks = iter(['CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node.servo_loop.step = lambda *a, **k: None
    node.servo_loop.start = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'closing'
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)
    node._get_current_tcp_posx = lambda: [0.0, 0.0, 50.0, 0.0, 0.0, 0.0]  # 고정 - 절대 안 오름

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert result.success is False
    assert '타임아웃' in result.message


# ---- handover_hold ----

def test_handover_hold_releases_on_pull_detected(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=1)])
    node._latest_robot_state = {'received_at': time.monotonic(), 'sample_seq': 1}
    # 이 테스트는 fresh-sample 필터링 자체가 아니라 "당김 감지 -> 개방" 배선을
    # 검증하는 것이므로, 새로 추가된 신선도 게이트는 항상 통과하도록 스텁한다
    # (신선도 자체는 별도 전용 테스트에서 검증한다).
    node._is_fresh_robot_state = lambda state, since, max_age: True
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: True
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_handover_hold(gh)

    assert calls == ['open']
    assert gh.succeeded is True
    assert result.success is True
    assert result.message == 'pull_detected, released'


def test_handover_hold_compliance_disable_failure_blocks_rg2_open(node, monkeypatch):
    """필수 테스트 5: pull 확정 후 compliance 해제 실패 시 RG2 open을 금지한다."""
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=1)])
    node._latest_robot_state = {'received_at': time.monotonic(), 'sample_seq': 1}
    node._is_fresh_robot_state = lambda state, since, max_age: True
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: (_ for _ in ()).throw(RuntimeError('compliance boom'))
    node._is_pull_detected = lambda state: True
    opened = []
    node.rg2_client.open = lambda goal_handle=None: opened.append(True)

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_handover_hold(gh)

    assert opened == []  # compliance 해제 실패 시 RG2를 열지 않는다
    assert gh.succeeded is False
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


def test_handover_hold_rg2_canceled_during_open_ends_as_canceled_not_fault(node, monkeypatch):
    # RG2 open의 busy 대기 "도중"에 취소가 확인된 경우(last_status=CANCELED)에도
    # FAULT가 아니라 기존 cleanup 경로(_finish_cancel)를 통해 canceled()로 끝나야 한다.
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=1)])
    node._latest_robot_state = {'received_at': time.monotonic(), 'sample_seq': 1}
    node._is_fresh_robot_state = lambda state, since, max_age: True
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: True

    def _fake_open(goal_handle=None):
        node.rg2_client.last_status = RG2Status.CANCELED
        return False

    node.rg2_client.open = _fake_open

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_handover_hold(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
    assert node.safety_state == SafetyState.NORMAL  # FAULT로 처리되지 않았다


def test_handover_hold_canceled_does_not_open_gripper(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._latest_robot_state = 'state'
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: False
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')

    gh = FakeGoalHandle(_goal('handover_hold'))
    gh.is_cancel_requested = True

    result = node._execute_handover_hold(gh)

    assert calls == []  # 취소 시 그리퍼를 열지 않는다 (낙하 방지)
    assert gh.was_canceled is True
    assert result.success is False


def test_handover_hold_fault_mid_loop_does_not_open_gripper(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._latest_robot_state = 'state'
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: False
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')
    node.safety_state = SafetyState.FAULT

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_handover_hold(gh)

    assert calls == []  # Fault 발생 시에도 그리퍼를 자동으로 열지 않는다
    assert gh.aborted is True
    assert result.success is False


def test_handover_hold_cancel_after_pull_confirmed_does_not_open_gripper(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=1)])
    node._is_fresh_robot_state = lambda state, since, max_age: True
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._latest_robot_state = {'received_at': time.monotonic(), 'sample_seq': 1}
    gh = FakeGoalHandle(_goal('handover_hold'))

    def is_pull_detected(state):
        # 당김이 확인된 직후, RG2를 열기 전에 취소가 들어온 상황을 시뮬레이션한다.
        gh.is_cancel_requested = True
        return True

    node._is_pull_detected = is_pull_detected
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')

    result = node._execute_handover_hold(gh)

    assert calls == []  # 그리퍼를 열지 않는다
    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False


def test_handover_hold_fault_after_pull_confirmed_does_not_open_gripper(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=1)])
    node._is_fresh_robot_state = lambda state, since, max_age: True
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._latest_robot_state = {'received_at': time.monotonic(), 'sample_seq': 1}
    gh = FakeGoalHandle(_goal('handover_hold'))

    def is_pull_detected(state):
        node.safety_state = SafetyState.FAULT
        return True

    node._is_pull_detected = is_pull_detected
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')

    result = node._execute_handover_hold(gh)

    assert calls == []
    assert gh.aborted is True
    assert gh.was_canceled is False
    assert result.success is False


def test_handover_hold_cancel_calls_real_move_stop_when_hardware_enabled(node):
    node.hardware_enabled = True
    fake = _FakeDoosanDriver()
    node._doosan = fake
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None

    gh = FakeGoalHandle(_goal('handover_hold'))
    gh.is_cancel_requested = True

    result = node._execute_handover_hold(gh)

    assert gh.was_canceled is True
    assert result.success is False
    assert fake.stop_calls == [node.get_parameter('safety.fault_stop_mode').value]


# ---- handover_hold: cleanup(MoveStop/compliance) 실패 시 취소 성공으로 가장하지 않음 ----

def test_handover_hold_cancel_aborts_as_fault_when_move_stop_returns_false(node):
    node.hardware_enabled = True
    fake = _FakeDoosanDriver()
    fake.stop_return_value = False
    node._doosan = fake
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')
    published = []
    node.pub_fault.publish = published.append

    gh = FakeGoalHandle(_goal('handover_hold'))
    gh.is_cancel_requested = True

    result = node._execute_handover_hold(gh)

    assert calls == []  # cleanup 실패 시에도 RG2를 자동으로 열지 않는다
    assert gh.was_canceled is False  # 취소 성공으로 가장하지 않는다
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT
    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.FAULT)


def test_handover_hold_cancel_aborts_as_fault_when_compliance_disable_raises(node):
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: (_ for _ in ()).throw(RuntimeError('compliance boom'))
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')

    gh = FakeGoalHandle(_goal('handover_hold'))
    gh.is_cancel_requested = True

    result = node._execute_handover_hold(gh)

    assert calls == []
    assert gh.was_canceled is False
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


def test_handover_hold_ignores_force_sample_older_than_hold_start(node, monkeypatch):
    import time as time_module

    node.set_parameters([
        Parameter('handover_hold.pull_axis_index', value=0),
        Parameter('handover_hold.pull_force_threshold_n', value=15.0),
        Parameter('handover_hold.pull_confirm_samples', value=1),
    ])
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    # handover_hold가 시작되기 전에 이미 당김 조건을 만족하는 '오래된' 샘플을 심어둔다.
    node._latest_robot_state = {
        'tool_force': [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'received_at': time.monotonic(),
    }

    gh = FakeGoalHandle(_goal('handover_hold'))
    poll_count = {'n': 0}

    def fake_sleep(_s):
        # 오래된 샘플은 계속 무시되어야 한다 - 몇 번 폴링한 뒤 취소로 테스트를 종료한다.
        poll_count['n'] += 1
        if poll_count['n'] >= 3:
            gh.is_cancel_requested = True

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)

    result = node._execute_handover_hold(gh)

    assert gh.was_canceled is True  # 오래된 샘플로 pull_detected 처리되지 않고 취소로 종료됨
    assert result.success is False


def _install_fake_monotonic_clock(monkeypatch, time_module):
    """time.monotonic()을 완전히 통제 가능한 가짜 시계로 대체한다.

    실제 wall-clock을 쓰면 테스트가 "샘플을 만든 시각"과 "hold_start_time을 기록하는
    시각" 사이의 극히 짧은(그러나 0은 아닌) 실제 시간차 때문에, 미리 만들어 둔 첫
    샘플이 항상 '시작 이전 샘플'로 오판되는 등 타이밍에 따라 결과가 달라지는 flaky한
    테스트가 된다. 그래서 sample_seq 관련 테스트들은 이 가짜 시계로 순서를 완전히
    고정한다."""
    clock = {'t': 0.0}
    monkeypatch.setattr(time_module, 'monotonic', lambda: clock['t'])
    return clock


def _make_incrementing_state_factory(clock):
    """호출할 때마다 시계를 전진시키고 새로운(distinct) sample_seq를 가진 신선한
    robot_state 딕셔너리를 만드는 factory. handover_hold 테스트에서 time.sleep mock을
    통해 매 폴링마다 '서로 다른 새 샘플'이 도착한 것처럼 시뮬레이션하기 위해 사용한다."""
    counter = {'n': 0}

    def _next_state():
        counter['n'] += 1
        clock['t'] += 1.0
        return {'received_at': clock['t'], 'sample_seq': counter['n']}

    return _next_state


def _install_hang_guard(gh, poll_count, limit=200):
    """production 로직에 회귀가 생겨 확정 조건에 절대 도달하지 못하더라도, 테스트
    스위트 전체가 무한 루프로 멈추지 않도록 하는 안전장치. 이 한도에 도달하면 강제로
    취소를 걸어 assertion 실패로 빠르게 끝나게 한다(행 방지 목적일 뿐 정상 동작에는
    영향을 주지 않아야 한다 - 각 테스트는 이 한도보다 훨씬 적은 반복 안에 끝나야 한다)."""
    poll_count['n'] += 1
    if poll_count['n'] > limit:
        gh.is_cancel_requested = True


def test_handover_hold_requires_consecutive_pull_samples(node, monkeypatch):
    import time as time_module

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=3)])
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    clock = _install_fake_monotonic_clock(monkeypatch, time_module)
    next_state = _make_incrementing_state_factory(clock)
    node._latest_robot_state = next_state()

    # True, True, False(연속이 끊겨 리셋), True, True, True 순서 - 서로 다른 새 샘플
    # 6개가 연속으로 도착했을 때만 6번째에서 확인된다 (요구사항 1의 필수 테스트 3).
    pattern = iter([True, True, False, True, True, True])
    node._is_pull_detected = lambda state: next(pattern)
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True

    gh = FakeGoalHandle(_goal('handover_hold'))
    poll_count = {'n': 0}

    def fake_sleep(_s):
        _install_hang_guard(gh, poll_count)
        node._latest_robot_state = next_state()

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)

    result = node._execute_handover_hold(gh)

    assert calls == ['open']
    assert result.success is True
    assert next(pattern, 'exhausted') == 'exhausted'  # 정확히 6개의 새 샘플 확인 후 종료됨


def test_handover_hold_distinct_new_samples_required_to_open(node, monkeypatch):
    """필수 테스트 2: 서로 다른 새 샘플 3개가 연속 true일 때만 개방한다."""
    import time as time_module

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=3)])
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    clock = _install_fake_monotonic_clock(monkeypatch, time_module)
    next_state = _make_incrementing_state_factory(clock)
    node._latest_robot_state = next_state()

    eval_count = {'n': 0}

    def is_pull_detected(state):
        eval_count['n'] += 1
        return True

    node._is_pull_detected = is_pull_detected
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True

    gh = FakeGoalHandle(_goal('handover_hold'))
    poll_count = {'n': 0}

    def fake_sleep(_s):
        _install_hang_guard(gh, poll_count)
        node._latest_robot_state = next_state()

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)

    result = node._execute_handover_hold(gh)

    assert calls == ['open']
    assert result.success is True
    assert eval_count['n'] == 3  # 정확히 서로 다른 새 샘플 3개만 평가되었다


def test_handover_hold_same_sample_seq_counted_only_once(node, monkeypatch):
    """필수 테스트 1: 같은 sample_seq의 당김 샘플을 여러 번 읽어도 평가는 1회뿐이다."""
    import time as time_module

    node.set_parameters([
        Parameter('handover_hold.pull_confirm_samples', value=2),
        # staleness가 아니라 순수하게 '같은 sample_seq 중복 조회' 자체만 검증하기
        # 위해, 신선도 임계값을 충분히 크게 두어 staleness가 끼어들지 않게 한다.
        Parameter('handover_hold.force_sample_max_age_s', value=1000.0),
    ])
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    clock = _install_fake_monotonic_clock(monkeypatch, time_module)
    node._latest_robot_state = {'received_at': clock['t'], 'sample_seq': 1}

    eval_count = {'n': 0}

    def is_pull_detected(state):
        eval_count['n'] += 1
        return True

    node._is_pull_detected = is_pull_detected
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open') or True

    gh = FakeGoalHandle(_goal('handover_hold'))
    poll_count = {'n': 0}

    def fake_sleep(_s):
        _install_hang_guard(gh, poll_count)
        clock['t'] += 1.0
        # 처음 10번의 폴링 동안은 같은 sample_seq(=1)를 그대로 반복해서 읽는다.
        if poll_count['n'] == 10:
            # 10번째 이후에만 새로운(distinct) 샘플이 도착한다.
            node._latest_robot_state = {'received_at': clock['t'], 'sample_seq': 2}

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)

    result = node._execute_handover_hold(gh)

    assert calls == ['open']
    assert result.success is True
    # sample_seq=1은 10번 읽혔지만 1회만 평가되었고(count=1에서 멈춤), sample_seq=2가
    # 도착한 뒤에야 두 번째 평가로 count=2가 되어 열린다 - 총 평가 횟수는 2회뿐이다.
    assert eval_count['n'] == 2


def test_handover_hold_stale_sample_does_not_confirm_pull(node, monkeypatch):
    """필수 테스트 5: stale(오래된) 샘플이 계속되면 pull이 확정되지 않는다."""
    import time as time_module

    node.set_parameters([
        Parameter('handover_hold.pull_confirm_samples', value=2),
        Parameter('handover_hold.force_sample_max_age_s', value=0.5),
    ])
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None

    clock = {'t': 0.0}
    monkeypatch.setattr(time_module, 'monotonic', lambda: clock['t'])

    # hold_start_time은 clock=0.0에서 기록된다. 샘플은 시작 직후(0.01)에 한 번
    # 신선하게 평가되어 count=1이 되지만, 그 뒤로는 시계만 흐르고(clock=1.0으로 도약)
    # robot_state 피드가 멈춘 것처럼 received_at이 갱신되지 않아 stale해진다.
    node._latest_robot_state = {'received_at': 0.01, 'sample_seq': 1}
    node._is_pull_detected = lambda state: True
    calls = []
    node.rg2_client.open = lambda goal_handle=None: calls.append('open')

    poll_count = {'n': 0}

    def fake_sleep(_s):
        poll_count['n'] += 1
        clock['t'] = 1.0  # max_age_s(0.5s)를 훌쩍 넘겨 이후 샘플을 stale로 만든다
        if poll_count['n'] >= 3:
            gh.is_cancel_requested = True

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)

    gh = FakeGoalHandle(_goal('handover_hold'))
    result = node._execute_handover_hold(gh)

    assert calls == []  # count가 0으로 초기화되어 확정되지 않고, 결국 취소로 종료된다
    assert gh.was_canceled is True
    assert result.success is False


def test_handover_hold_stale_sample_resets_pull_count_to_zero(node, monkeypatch):
    """필수 테스트 7: stale 샘플 발생 시 pull count가 0으로 초기화되어, 단절 이전의
    count가 단절 이후로 이어지지 않는다."""
    import time as time_module

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=2)])
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    clock = {'t': 0.0}
    monkeypatch.setattr(time_module, 'monotonic', lambda: clock['t'])

    node._latest_robot_state = {'received_at': 0.0, 'sample_seq': 1}
    evaluated_seqs = []

    def is_pull_detected(state):
        evaluated_seqs.append(state['sample_seq'])
        return True

    node._is_pull_detected = is_pull_detected
    opened = []
    node.rg2_client.open = lambda goal_handle=None: opened.append(True)

    gh = FakeGoalHandle(_goal('handover_hold'))
    step = {'n': 0}

    def fake_sleep(_s):
        step['n'] += 1
        if step['n'] == 1:
            # 센서 피드가 오래 끊긴 것처럼 시계를 크게 전진시켜 기존 샘플을 stale로
            # 만든다 - 이때 count(1)가 0으로 초기화되어야 한다.
            clock['t'] += 1000.0
        elif step['n'] == 2:
            # 단절 이후 새로운 fresh 샘플 하나만 도착한다 - count가 이어졌다면(2)
            # 즉시 확정되겠지만, 리셋되었다면 이번 한 번의 true로는 count=1이라
            # 아직 confirm_needed(2)에 도달하지 못한다.
            node._latest_robot_state = {'received_at': clock['t'], 'sample_seq': 2}
        else:
            gh.is_cancel_requested = True

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)

    result = node._execute_handover_hold(gh)

    assert opened == []  # count가 리셋되어 confirm_needed(2)에 도달하지 못하고 취소로 종료됨
    assert gh.was_canceled is True
    assert result.success is False
    assert evaluated_seqs == [1, 2]  # 단절 전 샘플 1회, 단절 후 새 샘플 1회만 평가됨


def test_dispatch_routes_handover_hold(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.set_parameters([Parameter('handover_hold.pull_confirm_samples', value=1)])
    node._latest_robot_state = {'received_at': time.monotonic(), 'sample_seq': 1}
    node._is_fresh_robot_state = lambda state, since, max_age: True
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: True
    node.rg2_client.open = lambda goal_handle=None: True

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


def _setup_servo_pick_dry(node):
    node.servo_loop.start = lambda *a, **k: None


def _setup_handover_hold_dry(node):
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None


@pytest.mark.parametrize('task_type,setup_fn,rg2_attr', [
    ('servo_pick', _setup_servo_pick_dry, 'close'),
    ('handover_hold', _setup_handover_hold_dry, 'open'),
])
def test_execute_aborts_without_rg2_action_when_rclpy_not_ok(
        node, monkeypatch, task_type, setup_fn, rg2_attr):
    """필수 테스트 6: rclpy.ok()가 False(프로세스 종료 중)이면 정상 close/pull
    조건으로 간주하지 않고, RG2 open/close를 호출하지 않은 채 abort한다."""
    monkeypatch.setattr(rclpy, 'ok', lambda: False)
    setup_fn(node)
    calls = []
    setattr(node.rg2_client, rg2_attr, lambda *a, **k: calls.append(True))

    gh = FakeGoalHandle(_goal(task_type))
    if task_type == 'servo_pick':
        gh.request.tool_class = 'spanner'
        gh.request.grasp_width_mm = 30.0
        gh.request.grasp_force_n = 20.0
        result = node._execute_servo_pick(gh)
    else:
        result = node._execute_handover_hold(gh)

    assert calls == []  # RG2 open/close가 호출되지 않는다
    assert gh.aborted is True
    assert gh.was_canceled is False
    assert result.success is False


# ---- 전달 방향 당김 판정 (다른 축의 힘/충돌을 당김으로 오판하지 않음) ----

def test_is_pull_detected_false_when_axis_unconfigured(node):
    state = {'tool_force': [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]}

    assert node._is_pull_detected(state) is False


def test_is_pull_detected_true_when_axis_configured_and_exceeds_threshold(node):
    node.set_parameters([
        Parameter('handover_hold.pull_axis_index', value=2),
        Parameter('handover_hold.pull_direction_sign', value=1),
        Parameter('handover_hold.pull_force_threshold_n', value=10.0),
    ])
    state = {'tool_force': [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]}

    assert node._is_pull_detected(state) is True


def test_is_pull_detected_false_for_other_axis_force(node):
    node.set_parameters([
        Parameter('handover_hold.pull_axis_index', value=2),
        Parameter('handover_hold.pull_force_threshold_n', value=10.0),
    ])
    state = {'tool_force': [50.0, 0.0, 0.0, 0.0, 0.0, 0.0]}  # x축의 큰 힘은 전달 방향이 아님

    assert node._is_pull_detected(state) is False


def test_is_pull_detected_rejects_moment_axis(node):
    # 3~5는 모멘트(Nm) 성분이라 힘 임계값(N)과 비교 대상이 아니다 - 허용하지 않는다.
    node.set_parameters([
        Parameter('handover_hold.pull_axis_index', value=4),
        Parameter('handover_hold.pull_force_threshold_n', value=1.0),
    ])
    state = {'tool_force': [0.0, 0.0, 0.0, 0.0, 100.0, 0.0]}

    assert node._is_pull_detected(state) is False


def test_default_handover_hold_ref_is_base(node):
    # GetToolForce.srv는 DR_TOOL(1)도 정의하지만 이 노드는 BASE(0)/WORLD(2)만 허용한다.
    assert node.get_parameter('handover_hold.ref').value == 0


def test_state_poll_timer_never_opens_gripper(node):
    opened = []
    node.rg2_client.open = lambda goal_handle=None: opened.append(True)
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.STANDBY, 'ext_torque': [0.0] * 6,
        'tool_force': [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]}

    node._on_state_poll_timer()

    assert opened == []


# ---- gripper / 상태 폴링 / Fault ----

def test_gripper_timer_publishes_state(node):
    from handover_interfaces.msg import GripperState

    node.rg2_client.get_state = lambda: (30.0, True)
    published = []
    node.pub_gripper_state.publish = published.append

    node._on_gripper_timer()

    assert len(published) == 1
    assert isinstance(published[0], GripperState)
    assert published[0].width_mm == 30.0
    assert published[0].grip_detected is True


def test_tf_broadcast_timer_does_nothing_when_posx_unavailable(node):
    # base_link -> link_6 TF는 이제 이 노드가 방송하지 않는다(dsr_bringup2의
    # robot_state_publisher가 담당) - GetCurrentPosx가 실패해도 TCP 캐시가
    # None으로 유지되는지만 확인한다.
    node.hardware_enabled = True

    class _FakeDoosan:
        def get_current_posx(self, ref=0):
            return None

    node._doosan = _FakeDoosan()

    node._on_tf_broadcast_timer()

    assert node._tcp_pose_cache is None


def test_state_poll_timer_silent_when_no_fault(node):
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.STANDBY, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert published == []


def test_state_poll_timer_skips_when_state_unavailable(node):
    node._read_robot_state = lambda: None
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert published == []
    assert node._latest_robot_state is None


def test_check_fault_detects_protective_stop(node):
    state = {'robot_state': DoosanRobotState.SAFE_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}

    reason = node._check_fault(state)

    assert reason.startswith(FaultPrefix.PROTECTIVE_STOP)


def test_check_fault_detects_emergency_stop(node):
    state = {'robot_state': DoosanRobotState.EMERGENCY_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}

    reason = node._check_fault(state)

    assert reason.startswith(FaultPrefix.EMERGENCY_STOP)


def test_check_fault_no_longer_reacts_to_external_torque(node):
    # 외력 감지는 이제 check_fault(ROS 서비스 폴링)가 아니라 DrflForceMonitor
    # (독립 쓰레드, DRFL 직접 연결)가 전담한다 - MOVING 중에도 동작해야 해서
    # 옮겼다(2026-07-06). ext_torque가 아무리 커도 STANDBY/robot_state 자체에
    # 문제가 없으면 check_fault는 항상 None을 반환해야 한다.
    state = {'robot_state': DoosanRobotState.STANDBY,
             'ext_torque': [0.0, 0.0, 999.0, 0.0, 0.0, 0.0], 'tool_force': [0.0] * 6}

    reason = node._check_fault(state)

    assert reason is None


def test_state_poll_timer_publishes_fault_when_detected(node):
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.EMERGENCY_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.EMERGENCY_STOP)
    assert node.safety_state == SafetyState.EMERGENCY_STOP


# ---- Fault 단계 상승(escalation) 및 E-Stop 지속 감시 ----
#
# 우선순위: NORMAL < PROTECTIVE_STOP < FAULT < EMERGENCY_STOP (robot_control에는
# task_manager의 RECOVERY_REQUIRED가 없다). 이미 비정상 상태여도 폴링 자체는 멈추지
# 않고, 더 높은 단계가 감지되면 즉시 반영해야 한다 (동일 메시지 반복만 dedup한다).

def test_state_poll_timer_escalates_protective_stop_to_emergency_stop(node):
    node.safety_state = SafetyState.PROTECTIVE_STOP
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.EMERGENCY_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert node.safety_state == SafetyState.EMERGENCY_STOP
    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.EMERGENCY_STOP)


def test_state_poll_timer_escalates_fault_to_emergency_stop(node):
    node.safety_state = SafetyState.FAULT
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.EMERGENCY_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert node.safety_state == SafetyState.EMERGENCY_STOP
    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.EMERGENCY_STOP)


def test_state_poll_timer_never_downgrades_emergency_stop(node):
    node.safety_state = SafetyState.EMERGENCY_STOP
    # 물리 E-Stop이 해제된 것처럼 보여도(SAFE_STOP만 감지), 소프트웨어가 자동으로
    # 강등해서는 안 된다 - 사용자의 명시적 /robot/recover 확인이 필요하다.
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.SAFE_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert node.safety_state == SafetyState.EMERGENCY_STOP
    assert published == []


def test_state_poll_timer_does_not_republish_identical_fault_message(node):
    node.safety_state = SafetyState.FAULT
    node._last_fault_reason = f'{FaultPrefix.FAULT}예상하지 못한 외력이 감지되었습니다 (ext_torque peak=25.0 Nm).'
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.STANDBY,
        'ext_torque': [0.0, 0.0, 25.0, 0.0, 0.0, 0.0], 'tool_force': [0.0] * 6}
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert published == []  # 완전히 동일한 메시지의 반복 발행 방지
    assert node.safety_state == SafetyState.FAULT


def test_state_poll_timer_keeps_polling_and_detects_estop_while_already_faulted(node):
    """비정상 상태(FAULT)에서도 상태 폴링 자체가 멈추지 않고, 물리 E-Stop처럼 더
    심각한 상태를 계속 감지할 수 있어야 한다."""
    node.safety_state = SafetyState.FAULT
    node._last_fault_reason = f'{FaultPrefix.FAULT}예상하지 못한 외력이 감지되었습니다 (ext_torque peak=25.0 Nm).'
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.EMERGENCY_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}

    node._on_state_poll_timer()

    assert node.safety_state == SafetyState.EMERGENCY_STOP
    assert node._latest_robot_state is not None  # 폴링 자체는 계속 수행됨


# ---- /robot/recover ----

def test_recover_already_normal(node):
    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is True


def test_recover_dry_run_clears_software_fault(node):
    node.safety_state = SafetyState.FAULT

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is True
    assert node.safety_state == SafetyState.NORMAL


def test_recover_dry_run_rejects_emergency_stop(node):
    node.safety_state = SafetyState.EMERGENCY_STOP

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert node.safety_state == SafetyState.EMERGENCY_STOP


# ---- /robot/recover (hardware_enabled=true): SetRobotControl success만으로 NORMAL 판단 금지 ----

def test_recover_hardware_requires_standby_confirmation_not_just_success(node):
    node.hardware_enabled = True
    node.safety_state = SafetyState.PROTECTIVE_STOP
    fake = _FakeDoosanDriver()
    # 첫 조회: SAFE_STOP -> CONTROL_RESET_SAFET_STOP 호출 -> 재조회: 아직 MOVING(복구 안 됨)
    fake.robot_state_sequence = [DoosanRobotState.SAFE_STOP, DoosanRobotState.MOVING]
    node._doosan = fake

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert node.safety_state == SafetyState.PROTECTIVE_STOP  # NORMAL로 바뀌지 않음
    assert fake.set_robot_control_calls == [DoosanRobotControl.CONTROL_RESET_SAFET_STOP]


def test_recover_hardware_succeeds_when_standby_confirmed(node):
    node.hardware_enabled = True
    node.safety_state = SafetyState.PROTECTIVE_STOP
    fake = _FakeDoosanDriver()
    fake.robot_state_sequence = [DoosanRobotState.SAFE_STOP, DoosanRobotState.STANDBY]
    node._doosan = fake

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is True
    assert node.safety_state == SafetyState.NORMAL


def test_recover_safe_stop2_reaching_recovery_does_not_clear_to_normal(node):
    node.hardware_enabled = True
    node.safety_state = SafetyState.PROTECTIVE_STOP
    fake = _FakeDoosanDriver()
    # SAFE_STOP2 -> CONTROL_RECOVERY_SAFE_STOP -> RECOVERY (STANDBY 아님, 추가 단계 필요)
    fake.robot_state_sequence = [DoosanRobotState.SAFE_STOP2, DoosanRobotState.RECOVERY]
    node._doosan = fake

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert node.safety_state == SafetyState.PROTECTIVE_STOP
    assert 'RECOVERY' in response.message
    assert fake.set_robot_control_calls == [DoosanRobotControl.CONTROL_RECOVERY_SAFE_STOP]


def test_recover_hardware_rejects_emergency_stop(node):
    node.hardware_enabled = True
    node.safety_state = SafetyState.EMERGENCY_STOP
    fake = _FakeDoosanDriver()
    fake.robot_state_sequence = [DoosanRobotState.EMERGENCY_STOP]
    node._doosan = fake

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert node.safety_state == SafetyState.EMERGENCY_STOP
    assert fake.set_robot_control_calls == []


# ---- /robot/recover: STANDBY 확인 후 외력(관절별 절대 임계값) 재검사 ----
#
# 예전엔 baseline/delta 방식이었지만, 외력 감지 전체를 DrflForceMonitor(절대
# 임계값 + 히스테리시스)로 옮기면서 recover()도 같은 direct_threshold_nm을
# 재사용하는 단순한 절대값 비교로 바꿨다(2026-07-06).

def test_recover_standby_succeeds_when_torque_within_threshold(node):
    node.hardware_enabled = True
    node.safety_state = SafetyState.FAULT
    fake = _FakeDoosanDriver()
    fake.robot_state_sequence = [DoosanRobotState.STANDBY]
    fake.ext_torque = [0.0] * 6  # 전부 임계값 이내
    node._doosan = fake

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is True
    assert node.safety_state == SafetyState.NORMAL


def test_recover_standby_blocked_when_torque_exceeds_threshold(node):
    node.hardware_enabled = True
    node.safety_state = SafetyState.FAULT
    fake = _FakeDoosanDriver()
    fake.robot_state_sequence = [DoosanRobotState.STANDBY]
    # 손목 관절(인덱스 5, 기본 threshold=10.0)이 여전히 초과
    fake.ext_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 15.0]
    node._doosan = fake

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert node.safety_state == SafetyState.FAULT


def test_recover_standby_blocked_when_torque_measurement_fails(node):
    node.hardware_enabled = True
    node.safety_state = SafetyState.FAULT
    fake = _FakeDoosanDriver()
    fake.robot_state_sequence = [DoosanRobotState.STANDBY]
    fake.ext_torque = None
    node._doosan = fake

    response = node._on_recover(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert node.safety_state == SafetyState.FAULT


# ---- hardware_enabled=true인데 DoosanDriver 초기화가 실패하는 경우 ----

def test_doosan_driver_init_failure_sets_fault_and_rejects_goals(node, monkeypatch):
    node.hardware_enabled = True
    published = []
    node.pub_fault.publish = published.append

    def _raise_runtime_error(*args, **kwargs):
        raise RuntimeError('dsr_msgs2를 임포트할 수 없습니다 (simulated).')

    # 이 워크스테이션에는 dsr_msgs2가 PYTHONPATH에 이미 올라와 있어 실제 ImportError를
    # 재현할 수 없으므로, DoosanDriver 생성자를 직접 대체해 초기화 실패를 시뮬레이션한다.
    monkeypatch.setattr(
        'robot_control.robot_control_node.DoosanDriver', _raise_runtime_error)

    node._init_doosan_driver()

    assert node._doosan is None
    assert node.safety_state == SafetyState.FAULT
    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.FAULT)

    response = node._goal_callback(_goal('move_named', named_target='watch'))
    assert response == GoalResponse.REJECT


# ---- hardware_enabled=false: 실제 하드웨어 함수가 호출되지 않음 ----

def test_scipy_and_pymodbus_are_not_imported_at_module_level():
    import robot_control.robot_control_node as rcn

    assert 'scipy' not in dir(rcn)
    assert 'pymodbus' not in dir(rcn)


def test_hardware_disabled_by_default(node):
    assert node.hardware_enabled is False
    assert node._doosan is None


def test_hardware_disabled_rg2_client_never_touches_modbus(node):
    assert node.rg2_client.hardware_enabled is False

    node.rg2_client.open()

    assert node.rg2_client._client is None



# ---- SpeedlWatchdog 통합 (명령이 끊기면 자동으로 vel=0 발행) ----

def test_servo_pick_watchdog_publishes_zero_when_no_command_computed(node):
    """워치독 통합 테스트: hardware_ready 상태에서 step()이 계속 None을 반환해
    (tcp pose 미확보 등) pet()이 호출되지 않으면, watchdog_timeout_s 이내에
    워치독이 자동으로 vel=0 SpeedlStream을 발행한다(2026-07-07 실측: 단일
    정지 명령으로 충분함을 확인)."""
    node.hardware_enabled = True
    node.set_parameters([
        Parameter('servo_pick.hardware_ready', value=True),
        Parameter('servo_pick.watchdog_timeout_s', value=0.05),
        Parameter('servo_pick.control_period_s', value=0.01),
        # 이 테스트는 xy 추적 단계의 워치독 동작만 검증한다 - TCP 위치 캐시를
        # 아예 None으로 고정해 두었으므로 들어올림 단계(_run_grasp_lift)는 시작
        # 자체가 안 되고 FAULT로 끝나버린다. 이 테스트와 무관하므로 꺼둔다.
        Parameter('servo_pick.lift_height_m', value=0.0),
    ])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    node.servo_loop.start = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'tracking'
    node._get_current_tcp_posx = lambda: None  # step()이 항상 None을 반환하게 함

    started = time.monotonic()

    def _tick():
        # 워치독이 발동할 시간을 벌어준다(0.05s timeout보다 넉넉하게), 이후 종료.
        if time.monotonic() - started < 0.4:
            return ('CONTINUE', None)
        return ('CLOSE', None)

    node._servo_pick_tick = _tick
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert result.success is True
    assert any(
        cmd.vx == 0.0 and cmd.vy == 0.0 and cmd.vz == 0.0
        for cmd in fake.publish_calls)  # 워치독이 최소 1회 vel=0을 발행했다


# ---- goal_sent 체크포인트 (파이프라인 점검.md 대응) ----

_GOAL_SENT_CHECKPOINT_CASES = [
    ('move_named', 'watch', 'B', 'move_watch_goal_sent'),
    ('move_named', 'handover_safe', 'F', 'handover_safe_goal_sent'),
    ('move_named', 'home', 'J', 'home_goal_sent'),
    ('handover_approach', '', 'H', 'handover_approach_goal_sent'),
    ('handover_hold', '', 'I', 'handover_hold_goal_sent'),
]


@pytest.mark.parametrize(
    'task_type,named_target,expected_phase,expected_checkpoint', _GOAL_SENT_CHECKPOINT_CASES)
def test_goal_callback_publishes_checkpoint_on_accept(
        node, task_type, named_target, expected_phase, expected_checkpoint):
    published = []
    node.pub_debug_events.publish = published.append

    response = node._goal_callback(_goal(task_type, named_target=named_target))

    assert response == GoalResponse.ACCEPT
    payload = json.loads(published[-1].data)
    assert payload['phase'] == expected_phase
    assert payload['checkpoint_id'] == expected_checkpoint
    assert payload['status'] == 'PASS'


def test_goal_callback_publishes_fail_checkpoint_on_reject(node):
    node.safety_monitor.state = SafetyState.FAULT
    published = []
    node.pub_debug_events.publish = published.append

    response = node._goal_callback(_goal('move_named', named_target='watch'))

    assert response == GoalResponse.REJECT
    payload = json.loads(published[-1].data)
    assert payload['checkpoint_id'] == 'move_watch_goal_sent'
    assert payload['status'] == 'FAIL'


def test_goal_callback_unmapped_target_does_not_publish_checkpoint(node):
    published = []
    node.pub_debug_events.publish = published.append

    node._goal_callback(_goal('move_named', named_target='front'))

    assert published == []


# ---- servo_pick/handover_hold 체크포인트 (파이프라인 점검.md 대응, Task 6) ----

class _FakeRg2Client:
    def __init__(self, close_ok=True, open_ok=True):
        self.close_ok = close_ok
        self.open_ok = open_ok
        self.last_status = RG2Status.SUCCESS
        self.close_calls = []
        self.open_calls = []

    def close(self, width_mm, force_n, goal_handle=None):
        self.close_calls.append((width_mm, force_n))
        if not self.close_ok:
            self.last_status = RG2Status.COMMUNICATION_ERROR
        return self.close_ok

    def open(self, goal_handle=None):
        self.open_calls.append(1)
        if not self.open_ok:
            self.last_status = RG2Status.COMMUNICATION_ERROR
        return self.open_ok

    def get_state(self):
        return (20.0, True)


def _fake_run_rt_tracking_with_close(node):
    """_run_rt_tracking을 통째로 우회하는 체크포인트 테스트용 스텁.

    실제로는 RG2 close가 _servo_pick_tick_with_grasp_lock을 통해 백그라운드
    스레드로 트래킹 루프 "안에서" 실행되므로, 여기서도 그 스레드가 이미 끝난
    것처럼 동일하게 흉내낸다 - 그래야 _execute_servo_pick의 이후 로직(join,
    _servo_pick_close_success 확인, checkpoint 발행)이 실제 경로와 같게 동작한다."""
    import threading

    def _fake(goal_handle, **kw):
        node._run_servo_pick_close(goal_handle, goal_handle.request)
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        node._servo_pick_close_thread = thread
        return ('ARRIVED', '')
    return _fake


def test_servo_pick_close_success_publishes_gripper_closed_pass(node, monkeypatch):
    node.rg2_client = _FakeRg2Client(close_ok=True)
    monkeypatch.setattr(
        node, '_run_rt_tracking', _fake_run_rt_tracking_with_close(node))
    published = []
    node.pub_debug_events.publish = published.append
    goal_handle = FakeGoalHandle(_goal('servo_pick'))
    goal_handle.request.grasp_width_mm = 20.0
    goal_handle.request.grasp_force_n = 10.0
    goal_handle.request.tool_class = 'spanner'

    node._execute_servo_pick(goal_handle)

    payloads = [json.loads(p.data) for p in published]
    matches = [p for p in payloads if p['checkpoint_id'] == 'gripper_closed']
    assert len(matches) == 1
    assert matches[0]['phase'] == 'D'
    assert matches[0]['status'] == 'PASS'


def test_servo_pick_close_failure_publishes_gripper_closed_fail(node, monkeypatch):
    node.rg2_client = _FakeRg2Client(close_ok=False)
    monkeypatch.setattr(
        node, '_run_rt_tracking', _fake_run_rt_tracking_with_close(node))
    published = []
    node.pub_debug_events.publish = published.append
    goal_handle = FakeGoalHandle(_goal('servo_pick'))
    goal_handle.request.grasp_width_mm = 20.0
    goal_handle.request.grasp_force_n = 10.0
    goal_handle.request.tool_class = 'spanner'

    node._execute_servo_pick(goal_handle)

    payload = json.loads(published[-1].data)
    assert payload['checkpoint_id'] == 'gripper_closed'
    assert payload['status'] == 'FAIL'


def test_handover_hold_success_publishes_compliance_and_gripper_open_checkpoints(node, monkeypatch):
    node.rg2_client = _FakeRg2Client(open_ok=True)
    monkeypatch.setattr(node, '_enable_compliance', lambda: None)
    monkeypatch.setattr(node, '_disable_compliance', lambda: None)
    monkeypatch.setattr(
        node.safety_monitor, 'wait_for_pull', lambda *a, **k: 'PULLED')
    published = []
    node.pub_debug_events.publish = published.append
    goal_handle = FakeGoalHandle(_goal('handover_hold'))

    node._execute_handover_hold(goal_handle)

    payloads = [json.loads(p.data) for p in published]
    checkpoint_ids = [p['checkpoint_id'] for p in payloads]
    assert 'compliance_mode_active' in checkpoint_ids
    assert 'compliance_mode_ended' in checkpoint_ids
    assert 'gripper_opened_on_pull' in checkpoint_ids
    for payload in payloads:
        assert payload['phase'] == 'I'
        assert payload['status'] == 'PASS'


def test_handover_hold_open_failure_publishes_gripper_opened_on_pull_fail(node, monkeypatch):
    node.rg2_client = _FakeRg2Client(open_ok=False)
    monkeypatch.setattr(node, '_enable_compliance', lambda: None)
    monkeypatch.setattr(node, '_disable_compliance', lambda: None)
    monkeypatch.setattr(
        node.safety_monitor, 'wait_for_pull', lambda *a, **k: 'PULLED')
    published = []
    node.pub_debug_events.publish = published.append
    goal_handle = FakeGoalHandle(_goal('handover_hold'))

    node._execute_handover_hold(goal_handle)

    payloads = [json.loads(p.data) for p in published]
    matches = [p for p in payloads if p['checkpoint_id'] == 'gripper_opened_on_pull']
    assert matches[-1]['status'] == 'FAIL'


# ---- servo_pick 접촉 감지(DrflContactMonitor) 배선 ----

def test_on_contact_detected_sets_flag(node):
    assert node._contact_flag is False

    node._on_contact_detected(12.3, 5.0)

    assert node._contact_flag is True


def test_suspend_resume_drfl_contact_monitor_noop_without_hardware(node):
    # hardware_enabled=False(기본 dry_run)에서는 _drfl_contact_monitor가 None이므로
    # suspend/resume이 예외 없이 조용히 넘어가야 한다.
    node._resume_drfl_contact_monitor()
    node._suspend_drfl_contact_monitor()


def test_servo_pick_step_consumes_contact_flag_and_locks_z(node):
    node._get_current_tcp_posx = lambda: [0.0, 0.0, 200.0, 0.0, 0.0, 0.0]  # mm
    node.servo_loop.start('spanner', 30.0, 20.0)
    node.servo_loop.on_tool_track(FakeToolTrackForNode(0.0, 0.0, 0.0, 0.05))
    node.servo_loop.on_tool_track(FakeToolTrackForNode(0.02, 0.0, 0.0, 0.05))
    node._contact_flag = True

    node._servo_pick_step()

    assert node._contact_flag is False
    assert node.servo_loop._z_locked is True


def test_execute_servo_pick_resumes_and_suspends_contact_monitor(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    resume_calls = []
    suspend_calls = []
    node._resume_drfl_contact_monitor = lambda: resume_calls.append(True)
    node._suspend_drfl_contact_monitor = lambda: suspend_calls.append(True)

    ticks = iter(['CONTINUE', 'CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node.servo_loop.step = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (29.4, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    node._execute_servo_pick(gh)

    assert resume_calls == [True]
    assert suspend_calls == [True]
