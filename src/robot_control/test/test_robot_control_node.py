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
