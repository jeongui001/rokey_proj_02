import rclpy
import pytest
from geometry_msgs.msg import PoseStamped
from rclpy.action import CancelResponse, GoalResponse
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

from handover_interfaces.action import RobotTask
from robot_control.robot_control_node import (
    DoosanRobotControl, DoosanRobotState, FaultPrefix, RobotControlNode, SafetyState,
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


def _goal(task_type, named_target=''):
    g = RobotTask.Goal()
    g.task_type = task_type
    g.named_target = named_target
    return g


def _pose_goal():
    g = RobotTask.Goal()
    g.task_type = 'move_pose'
    g.target_pose = PoseStamped()
    return g


class _FakeDoosanDriver:
    """dsr_msgs2 없이 RobotControlNode의 오케스트레이션 로직만 검증하기 위한 가짜 드라이버."""

    def __init__(self):
        self.robot_state_sequence = []
        self.set_robot_control_calls = []
        self.ext_torque = [0.0] * 6
        self.tool_force = [0.0] * 6
        self.open_rt_session_should_fail = False
        self.publish_calls = []

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

    def open_rt_session(self):
        if self.open_rt_session_should_fail:
            raise RuntimeError('start_rt_control이 실패했습니다 (fake).')
        return True

    def close_rt_session(self):
        pass

    def stop(self, stop_mode=1):
        return True

    def publish_speedl_rt(self, cmd):
        self.publish_calls.append(cmd)


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


def test_move_named_unconfigured_named_pose_is_treated_as_failure(node):
    # 기본 named_poses.*는 빈 리스트이므로 실제 이동을 시도하지 않고 실패해야 한다.
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


def test_release_and_retry_calls_open_and_move_to_watch(node):
    calls = []
    node.rg2_client.open = lambda: calls.append('open')
    node._call_move_service = lambda **kw: calls.append(('move', kw)) or True
    gh = FakeGoalHandle(_goal('release_and_retry'))

    result = node._execute_release_and_retry(gh)

    assert calls[0] == 'open'
    assert calls[1] == ('move', {'named_target': 'watch', 'goal_handle': gh})
    assert gh.succeeded is True
    assert result.success is True


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


def test_dispatch_routes_place_down_to_move_named_handler(node):
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_goal('place_down', named_target='place_down'))

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


# ---- move_pose ----

def test_move_pose_success(node):
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_pose_goal())

    result = node._execute_move_pose(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_move_pose_failure(node):
    node._call_move_service = lambda **kw: False
    gh = FakeGoalHandle(_pose_goal())

    result = node._execute_move_pose(gh)

    assert gh.aborted is True
    assert result.success is False


def test_move_pose_canceled_calls_canceled_not_abort(node):
    node._call_move_service = lambda **kw: False
    gh = FakeGoalHandle(_pose_goal())
    gh.is_cancel_requested = True

    result = node._execute_move_pose(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False


def test_dispatch_routes_move_pose(node):
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_pose_goal())

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


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


def test_execute_servo_pick_success_closes_gripper_and_returns_result(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    ticks = iter(['CONTINUE', 'CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node.servo_loop.step = lambda: None
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: None
    node._estimate_payload = lambda: 0.31
    node.rg2_client.close = lambda width, force: None
    node.rg2_client.get_state = lambda: (29.4, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.succeeded is True
    assert result.success is True
    assert result.measured_payload_kg == 0.31
    assert result.final_width_mm == 29.4
    assert result.grip_detected is True
    assert len(gh.feedback_msgs) == 3


def test_execute_servo_pick_abort_returns_reason(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._servo_pick_tick = lambda: ('ABORT', 'diverged')
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'diverged'


def test_execute_servo_pick_cancel_mid_loop_calls_canceled_and_closes_rt_session(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    rt_closed = []
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: rt_closed.append(True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
    assert rt_closed == [True]  # finally에서 RT 세션 정리됨


def test_servo_pick_aborts_on_tracking_loss(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.servo_loop = ServoLoop(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.1,
                                 eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                                 dt_latency=0.05, timeout_s=5.0, t_lost_s=0.0)
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'lost'


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
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: None
    node._estimate_payload = lambda: 0.3
    node.rg2_client.close = lambda width, force: None
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
    node.rg2_client.close = lambda width, force: None
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_servo_pick_aborts_when_start_rt_control_fails(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    fake.open_rt_session_should_fail = True
    node._doosan = fake
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert fake.publish_calls == []  # RT가 시작되지 않았으므로 속도 명령을 보내지 않는다


def test_servo_pick_publishes_speedl_only_when_hardware_ready_and_rt_confirmed(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    ticks = iter(['CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node.servo_loop.step = lambda: ServoCommand(vx=0.1)
    node.servo_loop.start = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'closing'
    node.rg2_client.close = lambda width, force: None
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert result.success is True
    assert len(fake.publish_calls) == 1


# ---- handover_hold ----

def test_handover_hold_releases_on_pull_detected(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._latest_robot_state = 'state'
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: True
    calls = []
    node.rg2_client.open = lambda: calls.append('open')

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_handover_hold(gh)

    assert calls == ['open']
    assert gh.succeeded is True
    assert result.success is True
    assert result.message == 'pull_detected, released'


def test_handover_hold_canceled_does_not_open_gripper(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._latest_robot_state = 'state'
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: False
    calls = []
    node.rg2_client.open = lambda: calls.append('open')

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
    node.rg2_client.open = lambda: calls.append('open')
    node.safety_state = SafetyState.FAULT

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_handover_hold(gh)

    assert calls == []  # Fault 발생 시에도 그리퍼를 자동으로 열지 않는다
    assert gh.aborted is True
    assert result.success is False


def test_dispatch_routes_handover_hold(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._latest_robot_state = 'state'
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: True
    node.rg2_client.open = lambda: None

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


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
    node.rg2_client.open = lambda: opened.append(True)
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


def test_check_fault_detects_unexpected_external_torque(node):
    state = {'robot_state': DoosanRobotState.STANDBY,
             'ext_torque': [0.0, 0.0, 25.0, 0.0, 0.0, 0.0], 'tool_force': [0.0] * 6}

    reason = node._check_fault(state)

    assert reason.startswith(FaultPrefix.FAULT)


def test_state_poll_timer_publishes_fault_when_detected(node):
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.EMERGENCY_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert len(published) == 1
    assert published[0].data.startswith(FaultPrefix.EMERGENCY_STOP)
    assert node.safety_state == SafetyState.EMERGENCY_STOP


def test_state_poll_timer_does_not_redeclare_once_faulted(node):
    node.safety_state = SafetyState.FAULT
    node._read_robot_state = lambda: {
        'robot_state': DoosanRobotState.EMERGENCY_STOP, 'ext_torque': [0.0] * 6, 'tool_force': [0.0] * 6}
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert published == []
    assert node.safety_state == SafetyState.FAULT


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


def test_hardware_disabled_rt_session_is_noop(node):
    node._open_rt_session()
    node._close_rt_session()

    assert node._doosan is None
