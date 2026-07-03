import rclpy
import pytest

from handover_interfaces.action import RobotTask
from robot_control.robot_control_node import RobotControlNode


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
        self.feedback_msgs = []

    def succeed(self):
        self.succeeded = True

    def abort(self):
        self.aborted = True

    def publish_feedback(self, fb):
        self.feedback_msgs.append(fb)


def _goal(task_type, named_target=''):
    g = RobotTask.Goal()
    g.task_type = task_type
    g.named_target = named_target
    return g


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


def test_move_named_stub_not_implemented_is_treated_as_failure(node):
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


def test_release_and_retry_calls_open_and_move_to_watch(node):
    calls = []
    node.rg2_client.open = lambda: calls.append('open')
    node._call_move_service = lambda **kw: calls.append(('move', kw)) or True
    gh = FakeGoalHandle(_goal('release_and_retry'))

    result = node._execute_release_and_retry(gh)

    assert calls[0] == 'open'
    assert calls[1] == ('move', {'named_target': 'watch'})
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


def test_state_poll_timer_publishes_fault_when_detected(node):
    node._read_robot_state = lambda: 'state'
    node._check_fault = lambda state: 'protective_stop'
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert len(published) == 1
    assert published[0].data == 'protective_stop'
    assert node._latest_robot_state == 'state'


def test_state_poll_timer_silent_when_no_fault(node):
    node._read_robot_state = lambda: 'state'
    node._check_fault = lambda state: None
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
