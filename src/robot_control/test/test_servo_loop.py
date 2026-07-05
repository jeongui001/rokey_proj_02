import time

import pytest
from handover_interfaces.msg import ToolTrack
from robot_control.servo_loop import ServoLoop, ServoState


def _make_loop(**overrides):
    kwargs = dict(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                  eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                  dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3)
    kwargs.update(overrides)
    return ServoLoop(**kwargs)


def _track(x=0.0, y=0.0, z=0.05, depth_valid=True):
    msg = ToolTrack()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    msg.depth_valid = depth_valid
    msg.approaching = True
    return msg


def test_initial_state_is_tracking():
    loop = _make_loop()
    assert loop.get_state() == ServoState.TRACKING


def test_start_resets_internal_state():
    loop = _make_loop()
    loop.on_tool_track(_track(x=0.1, y=0.1))

    loop.start('spanner', 30.0, 20.0)

    assert loop.get_state() == ServoState.TRACKING
    assert loop.should_close() is False
    assert loop.should_abort() is None


def test_step_with_no_track_yet_returns_zero_command():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)

    cmd = loop.step()

    assert cmd.vx == 0.0 and cmd.vy == 0.0 and cmd.vz == 0.0 and cmd.yaw_rate == 0.0


def test_step_commands_velocity_opposite_to_xy_error():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=0.1, y=-0.05, z=0.05))

    cmd = loop.step()

    assert cmd.vx < 0.0  # 오차(+x)를 줄이는 방향
    assert cmd.vy > 0.0  # 오차(-y)를 줄이는 방향
    assert cmd.vz == 0.0  # xy 오차가 커서 아직 하강하지 않음


def test_step_clips_velocity_to_v_max():
    loop = _make_loop(v_max=0.2)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=10.0, y=0.0, z=0.05))

    cmd = loop.step()

    assert cmd.vx == pytest.approx(-0.2)


def test_step_descends_once_xy_error_small():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=0.001, y=0.001, z=0.05))

    cmd = loop.step()

    assert cmd.vz == pytest.approx(-0.10)


def test_on_tool_track_transitions_to_closing_when_near_target():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)

    loop.on_tool_track(_track(x=0.001, y=0.001, z=0.005))

    assert loop.get_state() == ServoState.CLOSING


def test_should_close_after_n_stable_cycles_within_grasp_tolerance():
    loop = _make_loop(n_stable=3, eps_grasp=0.005, z_close_m=0.01)
    loop.start('spanner', 30.0, 20.0)

    for _ in range(2):
        loop.on_tool_track(_track(x=0.001, y=0.001, z=0.005))
        assert loop.should_close() is False

    loop.on_tool_track(_track(x=0.001, y=0.001, z=0.005))

    assert loop.should_close() is True


def test_should_close_false_if_z_gap_still_large():
    loop = _make_loop(n_stable=1, eps_grasp=0.005, z_close_m=0.01)
    loop.start('spanner', 30.0, 20.0)

    loop.on_tool_track(_track(x=0.001, y=0.001, z=0.05))

    assert loop.should_close() is False


def test_should_abort_timeout():
    loop = _make_loop(timeout_s=0.01)
    loop.start('spanner', 30.0, 20.0)
    time.sleep(0.02)

    assert loop.should_abort() == 'timeout'


def test_should_abort_lost_when_no_update_within_t_lost():
    loop = _make_loop(timeout_s=5.0, t_lost_s=0.01)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track())
    time.sleep(0.02)

    assert loop.should_abort() == 'lost'


def test_should_abort_lost_when_depth_invalid():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)

    loop.on_tool_track(_track(depth_valid=False))

    assert loop.should_abort() == 'lost'


def test_should_abort_diverged_when_error_grows():
    loop = _make_loop(diverge_window=3, diverge_factor=1.2)
    loop.start('spanner', 30.0, 20.0)

    loop.on_tool_track(_track(x=0.01, y=0.0))
    loop.on_tool_track(_track(x=0.02, y=0.0))
    loop.on_tool_track(_track(x=0.05, y=0.0))

    assert loop.should_abort() == 'diverged'


def test_should_abort_none_when_converging():
    loop = _make_loop(diverge_window=3, diverge_factor=1.2, timeout_s=5.0, t_lost_s=5.0)
    loop.start('spanner', 30.0, 20.0)

    loop.on_tool_track(_track(x=0.05, y=0.0))
    loop.on_tool_track(_track(x=0.03, y=0.0))
    loop.on_tool_track(_track(x=0.01, y=0.0))

    assert loop.should_abort() is None
