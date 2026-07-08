import rclpy
import pytest

from robot_control.doosan_driver import DoosanDriver
from robot_control.robot_control_node import RobotControlNode
from robot_control.servo_loop import ServoCommand


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


def test_publish_speedl_converts_mps_to_mmps(node):
    """ServoLoop(m/s 단위)와 SpeedlStream.vel(mm/s 단위, tools/probe_speedl_stream.py로
    실기 확인됨) 사이 단위 변환이 publish_speedl에서 이뤄지는지 검증한다."""
    driver = DoosanDriver(node)
    published = []
    driver._pub_speedl.publish = published.append

    command = ServoCommand(vx=0.25, vy=-0.10, vz=0.05, yaw_rate=0.0)
    driver.publish_speedl(
        command, accel_param_prefix='servo_pick', period_param_name='servo_pick.control_period_s')

    assert len(published) == 1
    msg = published[0]
    assert msg.vel[0] == pytest.approx(250.0)
    assert msg.vel[1] == pytest.approx(-100.0)
    assert msg.vel[2] == pytest.approx(50.0)
    assert msg.vel[3] == 0.0 and msg.vel[4] == 0.0 and msg.vel[5] == 0.0
