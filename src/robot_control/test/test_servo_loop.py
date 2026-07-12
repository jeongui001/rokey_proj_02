import math
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
                  z_close=0.02, diverge_n=5, cov_threshold=0.5,
                  v_grasp_max=1.0, n_stable_v=1)
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
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05, n_stable_z=2, cov_threshold=2.5)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop.should_close() is True
    assert loop.get_state() == ServoState.CLOSING


def test_grasp_locked_keeps_vz_zero_and_xy_tracking_after_should_close():
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05, n_stable_z=2, cov_threshold=2.5)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop.should_close() is True

    # 그리퍼가 닫히는 동안 z_gap이 다시 커져도(원래대로면 DESCENDING으로 되돌아가
    # 재하강했을 상황) vz는 0으로 영구히 잠겨 있어야 하고, x,y는 계속 추적해야 한다.
    cmd = loop.step((0.1, 0.1, 0.20, 0, 0, 0), time.monotonic())
    assert cmd.vz == 0.0
    assert loop.get_state() == ServoState.CLOSING
    assert cmd.vx != 0.0 or cmd.vy != 0.0


def test_should_close_blocked_by_high_velocity_covariance():
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05, n_stable_z=2, cov_threshold=0.5)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop.should_close() is False


def test_should_close_blocked_by_high_tool_velocity():
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05, n_stable_z=2,
                       v_grasp_max=0.01, n_stable_v=2)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.5, 0.0, 0.05))  # 빠른 이동 -> 추정 속도 큼
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop.should_close() is False


def test_v_stable_count_requires_consecutive_low_velocity_ticks():
    loop = _make_loop(v_grasp_max=0.01, n_stable_v=3)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))  # 정지 물체 -> 추정 속도 ~0
    loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop._v_stable_count == 1
    loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop._v_stable_count == 2
    loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop._v_stable_count == 3


def test_single_noisy_z_gap_dip_does_not_latch_descent_stop():
    # z_close 이내로 딱 한 틱만 떨어졌다가 다시 벗어나는 경우(depth 노이즈 상황) -
    # n_stable_z가 2 이상이면 그 한 틱만으로 하강이 멈추면 안 된다.
    loop = _make_loop(z_close=0.05, n_stable_z=3)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.5))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.5))
    tcp_noisy_close = (0.0, 0.0, 0.48, 0, 0, 0)  # z_gap=0.02 < z_close(0.05), 노이즈성 단발 근접
    tcp_far = (0.0, 0.0, 0.20, 0, 0, 0)          # z_gap=0.30 >= z_close, 원래 거리로 복귀
    cmd = loop.step(tcp_noisy_close, time.monotonic())
    assert cmd.vz != 0.0
    assert loop._z_stable_count == 1
    cmd = loop.step(tcp_far, time.monotonic())
    assert loop._z_stable_count == 0
    assert cmd.vz != 0.0


def test_vz_locks_to_zero_only_after_n_stable_z_consecutive_ticks():
    loop = _make_loop(z_close=0.05, n_stable_z=3)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.5))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.5))
    tcp_close = (0.0, 0.0, 0.48, 0, 0, 0)  # z_gap=0.02 < z_close(0.05)로 계속 유지
    cmd = loop.step(tcp_close, time.monotonic())
    assert cmd.vz != 0.0  # 1번째 - 아직 안 멈춤
    cmd = loop.step(tcp_close, time.monotonic())
    assert cmd.vz != 0.0  # 2번째 - 아직 안 멈춤
    cmd = loop.step(tcp_close, time.monotonic())
    assert cmd.vz == 0.0  # 3번째(n_stable_z) - 이제 멈춤


def test_z_lock_persists_after_noisy_z_gap_reopens():
    # n_stable_z에 한 번 도달해 vz=0으로 멈추면, 이후 depth 노이즈로 z_gap이 다시
    # z_close 밖으로 벌어져 _z_stable_count가 리셋되더라도 재하강하지 않아야 한다.
    loop = _make_loop(z_close=0.05, n_stable_z=3, eps_descend=0.5)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.5))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.5))
    tcp_close = (0.0, 0.0, 0.48, 0, 0, 0)  # z_gap=0.02 < z_close(0.05)
    tcp_far = (0.0, 0.0, 0.20, 0, 0, 0)    # z_gap=0.30 >= z_close(0.05), 노이즈로 재이탈
    for _ in range(3):
        cmd = loop.step(tcp_close, time.monotonic())
    assert cmd.vz == 0.0
    assert loop._z_locked is True

    cmd = loop.step(tcp_far, time.monotonic())
    assert loop._z_stable_count == 0  # 라이브 카운트는 리셋되지만
    assert cmd.vz == 0.0              # 잠금 덕분에 재하강하지 않는다


def test_descend_vz_capped_by_stopping_distance_near_target():
    # z_gap이 작을 땐 descend_accel_m_s2로 정지 가능한 속도(sqrt(2*a*z_gap))로
    # vz가 캡핑되어야 한다 - 감속 없이 descend_speed 그대로 내려가다 명령만 0으로
    # 바뀌면 가속도 제한 때문에 관성으로 더 내려간다(2026-07-10 실기: 바닥 충돌).
    loop = _make_loop(eps_descend=0.5, z_close=0.001, n_stable_z=100,
                       descend_accel_m_s2=0.1)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.02))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.02))
    tcp = (0.0, 0.0, 0.03, 0, 0, 0)
    cmd = loop.step(tcp, time.monotonic())
    expected_speed = math.sqrt(2.0 * 0.1 * loop._last_z_gap)
    assert cmd.vz == pytest.approx(-expected_speed, rel=1e-3)
    assert abs(cmd.vz) < loop.descend_speed


def test_descend_vz_capped_using_margin_shifted_brake_distance():
    # descend_stop_margin_m>0이면 제동 곡선이 z_gap=0이 아니라 z_gap=margin에서
    # 속도 0을 겨냥해야 한다 - sqrt(2*a*max(z_gap-margin,0)).
    loop = _make_loop(eps_descend=0.5, z_close=0.001, n_stable_z=100,
                       descend_accel_m_s2=0.1, descend_stop_margin_m=0.01)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.02))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.02))
    tcp = (0.0, 0.0, 0.08, 0, 0, 0)
    cmd = loop.step(tcp, time.monotonic())
    expected_speed = math.sqrt(2.0 * 0.1 * max(loop._last_z_gap - 0.01, 0.0))
    assert cmd.vz == pytest.approx(-expected_speed, rel=1e-3)
    assert abs(cmd.vz) > 0.0


def test_descend_vz_is_zero_when_z_gap_within_margin():
    # z_gap이 margin 이내로 좁혀지면(제동 곡선상 남은 제동거리가 0) vz가 0이어야
    # 한다 - 표면 접촉 없이 margin 지점에서 자연스럽게 멈춰야 한다.
    loop = _make_loop(eps_descend=0.5, z_close=0.001, n_stable_z=100,
                       descend_accel_m_s2=0.1, descend_stop_margin_m=0.01)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.02))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.02))
    tcp = (0.0, 0.0, 0.025, 0, 0, 0)
    cmd = loop.step(tcp, time.monotonic())
    assert loop._last_z_gap <= 0.01
    assert cmd.vz == 0.0


def test_descend_vz_equals_descend_speed_when_far_from_target():
    # z_gap이 커서 정지 가능 속도가 descend_speed보다 크면, 예전처럼 descend_speed
    # 그대로 상한이 걸려야 한다(원거리 하강 속도는 그대로 유지).
    loop = _make_loop(eps_descend=0.5, descend_speed=0.10, descend_accel_m_s2=0.1)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.5))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.5))
    tcp = (0.0, 0.0, 0.3, 0, 0, 0)
    cmd = loop.step(tcp, time.monotonic())
    assert cmd.vz == pytest.approx(-0.10, abs=1e-6)


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


def test_should_abort_not_diverging_when_monotonic_increase_below_min_delta():
    # 5틱 연속 증가하지만 총 증가폭이 diverge_min_delta_m(기본 0.01m) 미만인 경우 -
    # 비전 갱신 사이 lead_time 외삽만으로 생기는 노이즈 수준 증가는 발산으로 보면 안 된다.
    loop = _make_loop(diverge_n=5)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05))
    loop._error_history = [0.100, 0.101, 0.102, 0.103, 0.104]
    assert loop.should_abort() is None


def test_should_abort_diverging_when_monotonic_increase_meets_min_delta():
    # 5틱 연속 증가하고 총 증가폭도 diverge_min_delta_m 이상이면 여전히 발산으로 잡아야 한다.
    loop = _make_loop(diverge_n=5)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05))
    loop._error_history = [0.10, 0.12, 0.14, 0.16, 0.20]
    assert loop.should_abort() == 'diverging'
