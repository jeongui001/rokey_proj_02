import time

import pytest
from geometry_msgs.msg import PoseStamped
from robot_control.servo_loop import (
    HandApproachServo,
    HandApproachState,
    ServoCommand,
    ServoLoop,
    ServoState,
)


class FakeHeader:
    def __init__(self, t):
        self.stamp = FakeStamp(t)


class FakeStamp:
    def __init__(self, t):
        self.sec = int(t)
        self.nanosec = int((t - int(t)) * 1e9)


class FakePosition:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakePose:
    def __init__(self, x, y, z):
        self.position = FakePosition(x, y, z)


class FakeToolTrack:
    def __init__(self, t, x, y, z, depth_valid=True):
        self.header = FakeHeader(t)
        self.pose = FakePose(x, y, z)
        self.depth_valid = depth_valid


def _make_loop(**overrides):
    kwargs = dict(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                  eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                  dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                  innov_low=0.010, innov_high=0.040, w_alpha=1.0,
                  z_close=0.02, diverge_n=5, cov_threshold=0.5)
    kwargs.update(overrides)
    return ServoLoop(**kwargs)


def _make_hand_servo(**overrides):
    kwargs = dict(kp_xy=1.0, v_max=0.15, timeout_s=5.0, t_lost_s=0.3, stop_distance_m=0.05)
    kwargs.update(overrides)
    return HandApproachServo(**kwargs)


def _hand_pose(x=0.0, y=0.0, z=0.0):
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    return msg


def test_initial_state_is_tracking():
    loop = _make_loop()
    assert loop.get_state() == ServoState.TRACKING


def test_step_before_any_track_returns_zero_command():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    cmd = loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert isinstance(cmd, ServoCommand)
    assert cmd.vx == 0.0 and cmd.vy == 0.0


def test_on_tool_track_then_step_moves_toward_target():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.80, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.78, 0.0, 0.05))
    cmd = loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert cmd.vx > 0.0


def test_step_respects_v_max():
    loop = _make_loop(v_max=0.05)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 1.0, 1.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 1.0, 1.0, 0.05))
    cmd = loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    speed = (cmd.vx ** 2 + cmd.vy ** 2) ** 0.5
    assert speed <= 0.05 + 1e-9


def test_large_innovation_resets_velocity_covariance_and_drops_w():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    trace_before = loop._filter.velocity_covariance_trace
    loop.on_tool_track(FakeToolTrack(0.04, 0.5, 0.0, 0.05))
    assert loop._w == pytest.approx(0.0, abs=1e-6)
    assert loop._filter.velocity_covariance_trace >= trace_before


def test_depth_invalid_track_does_not_move_z_estimate():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05, depth_valid=True))
    loop.on_tool_track(FakeToolTrack(0.02, 0.5, 0.0, 999.0, depth_valid=False))
    assert loop._filter.position[2] == pytest.approx(0.05, abs=1e-6)


def test_should_close_requires_stable_error_and_z_gap_and_low_covariance():
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05, cov_threshold=2.5)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop.should_close() is True
    assert loop.get_state() == ServoState.CLOSING


def test_should_close_blocked_by_high_velocity_covariance():
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05, cov_threshold=0.5)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop.should_close() is False


def test_should_abort_timeout():
    loop = _make_loop(timeout_s=0.0)
    loop.start('spanner', 30.0, 20.0)
    assert loop.should_abort() == 'timeout'


def test_should_abort_tracking_lost():
    loop = _make_loop(t_lost_s=0.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05))
    time.sleep(0.01)
    assert loop.should_abort() == 'tracking_lost'


def test_should_abort_none_when_healthy():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05))
    assert loop.should_abort() is None


# ==== HandApproachServo (handover_approach) ====

def test_hand_approach_initial_state_is_tracking():
    servo = _make_hand_servo()
    servo.start()

    assert servo.get_state() == HandApproachState.TRACKING


def test_hand_approach_step_with_no_pose_yet_returns_zero_command():
    servo = _make_hand_servo()
    servo.start()

    cmd = servo.step()

    assert cmd.vx == 0.0 and cmd.vy == 0.0 and cmd.vz == 0.0 and cmd.yaw_rate == 0.0


def test_hand_approach_step_commands_velocity_toward_positive_error():
    servo = _make_hand_servo()
    servo.start()
    servo.on_hand_pose(_hand_pose(x=0.1, y=0.05, z=0.02))

    cmd = servo.step()

    assert cmd.vx > 0.0
    assert cmd.vy > 0.0
    assert cmd.vz > 0.0
    assert cmd.yaw_rate == 0.0  # orientation 의미 미정의 - 항상 0


def test_hand_approach_step_commands_velocity_toward_negative_error():
    servo = _make_hand_servo()
    servo.start()
    servo.on_hand_pose(_hand_pose(x=-0.1, y=-0.05, z=-0.02))

    cmd = servo.step()

    assert cmd.vx < 0.0
    assert cmd.vy < 0.0
    assert cmd.vz < 0.0


def test_hand_approach_step_clips_velocity_to_v_max():
    servo = _make_hand_servo(v_max=0.2)
    servo.start()
    servo.on_hand_pose(_hand_pose(x=10.0, y=0.0, z=0.0))

    cmd = servo.step()

    assert cmd.vx == pytest.approx(0.2)


def test_hand_approach_should_stop_within_distance():
    servo = _make_hand_servo(stop_distance_m=0.05)
    servo.start()
    servo.on_hand_pose(_hand_pose(x=0.03, y=0.0, z=0.0))  # 3D 거리 0.03m < 0.05m

    assert servo.should_stop() is True
    assert servo.get_state() == HandApproachState.ARRIVED


def test_hand_approach_should_not_stop_outside_distance():
    servo = _make_hand_servo(stop_distance_m=0.05)
    servo.start()
    servo.on_hand_pose(_hand_pose(x=0.1, y=0.1, z=0.1))  # 3D 거리 > 0.05m

    assert servo.should_stop() is False
    assert servo.get_state() == HandApproachState.TRACKING


def test_hand_approach_should_stop_false_before_any_pose():
    servo = _make_hand_servo()
    servo.start()

    assert servo.should_stop() is False


def test_hand_approach_should_abort_timeout():
    servo = _make_hand_servo(timeout_s=0.01)
    servo.start()
    time.sleep(0.02)

    assert servo.should_abort() == 'timeout'


def test_hand_approach_should_abort_lost_when_no_update_within_t_lost():
    servo = _make_hand_servo(timeout_s=5.0, t_lost_s=0.01)
    servo.start()
    servo.on_hand_pose(_hand_pose(x=0.2, y=0.0, z=0.0))
    time.sleep(0.02)

    assert servo.should_abort() == 'lost'


def test_hand_approach_should_abort_diverged_when_error_grows():
    servo = _make_hand_servo(diverge_window=3, diverge_factor=1.2)
    servo.start()

    servo.on_hand_pose(_hand_pose(x=0.01, y=0.0, z=0.0))
    servo.on_hand_pose(_hand_pose(x=0.02, y=0.0, z=0.0))
    servo.on_hand_pose(_hand_pose(x=0.05, y=0.0, z=0.0))

    assert servo.should_abort() == 'diverged'


def test_hand_approach_should_abort_none_when_converging():
    servo = _make_hand_servo(diverge_window=3, diverge_factor=1.2, timeout_s=5.0, t_lost_s=5.0)
    servo.start()

    servo.on_hand_pose(_hand_pose(x=0.2, y=0.0, z=0.0))
    servo.on_hand_pose(_hand_pose(x=0.1, y=0.0, z=0.0))
    servo.on_hand_pose(_hand_pose(x=0.06, y=0.0, z=0.0))

    assert servo.should_abort() is None
