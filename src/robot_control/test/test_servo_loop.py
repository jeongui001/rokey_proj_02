import time

import numpy as np
import pytest
from robot_control.servo_loop import ServoCommand, ServoLoop, ServoState


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


def test_step_lead_time_includes_elapsed_since_last_track():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.02, 0.0, 0.05))
    t0 = loop._last_msg_time
    tcp = (0.0, 0.0, 0.05, 0, 0, 0)

    loop.step(tcp, t0)
    fresh_error = loop._last_e_xy_norm

    loop.step(tcp, t0 + 0.2)
    stale_error = loop._last_e_xy_norm

    fresh_p = loop._filter.predict_position(loop.dt_latency)
    stale_p = loop._filter.predict_position(loop.dt_latency + 0.2)
    expected_fresh = float(np.hypot(fresh_p[0] - tcp[0], fresh_p[1] - tcp[1]))
    expected_stale = float(np.hypot(stale_p[0] - tcp[0], stale_p[1] - tcp[1]))

    assert fresh_error == pytest.approx(expected_fresh, abs=1e-9)
    assert stale_error == pytest.approx(expected_stale, abs=1e-9)
    assert stale_error > fresh_error


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


def test_vy_locks_to_zero_once_y_error_small():
    loop = _make_loop(eps_y_close=0.01)
    loop.start('spanner', 30.0, 20.0)
    # x는 아직 멀고(0.80->0.78), y는 tcp_y(0.0)에 이미 충분히 가까움(0.005 < eps_y_close)
    loop.on_tool_track(FakeToolTrack(0.0, 0.80, 0.005, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.78, 0.005, 0.05))
    cmd = loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert cmd.vy == 0.0
    assert cmd.vx > 0.0


def test_vy_stays_locked_even_if_y_error_grows_again():
    loop = _make_loop(eps_y_close=0.01)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.80, 0.005, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.78, 0.005, 0.05))
    loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())

    # 이후 y 타깃이 다시 멀어져도(래치이므로) vy는 계속 0이어야 함
    for i in range(5):
        loop.on_tool_track(FakeToolTrack(0.04 + i * 0.02, 0.76 - i * 0.02, 0.5, 0.05))
        cmd = loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop._last_e_xy_norm > 0.01
    assert cmd.vy == 0.0
