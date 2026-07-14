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


class FakeQuaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class FakePose:
    def __init__(self, x, y, z, yaw_deg=None):
        self.position = FakePosition(x, y, z)
        if yaw_deg is None:
            self.orientation = FakeQuaternion()  # identity
        else:
            half = math.radians(yaw_deg) / 2.0
            self.orientation = FakeQuaternion(z=math.sin(half), w=math.cos(half))


class FakeToolTrack:
    def __init__(self, t, x, y, z, depth_valid=True, yaw_deg=None):
        self.header = FakeHeader(t)
        self.pose = FakePose(x, y, z, yaw_deg=yaw_deg)
        self.depth_valid = depth_valid
        self.yaw_valid = yaw_deg is not None


def _make_loop(**overrides):
    kwargs = dict(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                  eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                  dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                  innov_low=0.010, innov_high=0.040, w_alpha=1.0,
                  z_close=0.02, diverge_n=5, cov_threshold=0.5,
                  v_grasp_max=1.0, n_stable_v=1, v_tool_deadband_m_s=0.03)
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


def test_debug_snapshot_flags_margin_close_invariant():
    # descend_stop_margin_m >= z_close면 z_gap이 z_close 밑으로 내려가기도 전에
    # 제동 곡선이 이미 vz=0을 겨냥해 z_stable_count가 영원히 0에 머무는 교착이
    # 생긴다(2026-07-13 실기 확인 - 그리퍼가 절대 안 닫힘). 생성자에서 막지는
    # 않되(제동 곡선만 격리해서 보는 위 두 테스트처럼 일부러 이 조합을 쓰는
    # 정상적인 용법이 있음), debug_snapshot()에서 바로 눈에 띄어야 한다.
    ok_loop = _make_loop(z_close=0.02, descend_stop_margin_m=0.005)
    assert ok_loop.debug_snapshot()['z_close_margin_ok'] is True

    bad_loop = _make_loop(z_close=0.018, descend_stop_margin_m=0.030)
    assert bad_loop.debug_snapshot()['z_close_margin_ok'] is False


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


def test_feedforward_suppressed_below_tool_speed_deadband():
    # 정지 물체는 카메라 노이즈만으로도 v_tool이 완전히 0은 아니게 추정된다(실측
    # 노이즈 바닥 p95=10.6mm/s, max=23.2mm/s). v_tool_deadband_m_s보다 작으면
    # 피드포워드(w*v_tool) 기여를 억제해 순수 P제어(kp_xy*e)만 남겨야 한다 -
    # 그렇지 않으면 이 노이즈가 그대로 속도 명령으로 나가 로봇이 떨린다.
    loop = _make_loop(v_tool_deadband_m_s=0.03)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.80, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.8001, 0.0, 0.05))  # 미세한 움직임(노이즈 수준)
    tool_speed = float(np.hypot(*loop._filter.velocity[:2]))
    assert tool_speed < 0.03  # 전제 조건: deadband 아래

    tcp = (0.79, 0.0, 0.05, 0, 0, 0)  # v_max 클리핑을 피하려 p_ref 근처로 둠
    cmd = loop.step(tcp, loop._last_msg_time)

    p_ref = loop._filter.predict_position(loop.dt_latency)
    expected_p_only = loop.kp_xy * (p_ref[0] - tcp[0])
    assert cmd.vx == pytest.approx(expected_p_only, abs=1e-6)


def test_feedforward_full_above_tool_speed_deadband():
    # v_tool_deadband_m_s보다 충분히 빠르면(실제 이동) 기존 제어식(w*v_tool+kp_xy*e)이
    # 그대로 적용돼야 한다 - deadband가 정상 추적 반응성을 깎으면 안 된다.
    loop = _make_loop(v_tool_deadband_m_s=0.03)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    x = 0.0
    for i in range(1, 5):
        x += 0.002  # 0.1m/s로 등속 이동 - 몇 프레임 누적돼야 필터 속도 추정치가 수렴
        loop.on_tool_track(FakeToolTrack(0.02 * i, x, 0.0, 0.05))
    tool_speed = float(np.hypot(*loop._filter.velocity[:2]))
    assert tool_speed > 0.03  # 전제 조건: deadband 위

    tcp = (0.0, 0.0, 0.05, 0, 0, 0)
    cmd = loop.step(tcp, loop._last_msg_time)

    p_ref = loop._filter.predict_position(loop.dt_latency)
    v_tool = loop._filter.velocity
    expected_vx = loop._w * v_tool[0] + loop.kp_xy * (p_ref[0] - tcp[0])
    assert cmd.vx == pytest.approx(expected_vx, abs=1e-6)


def test_yaw_target_updates_from_valid_track_and_feeds_into_step_yaw_rate():
    # 탐지된 시점부터(첫 ToolTrack부터) yaw P 제어가 즉시 반영되는지 확인 -
    # "완전히 내려간 뒤 6축을 돌리는" 대신 서보잉과 동시에 회전 명령이 나가야 한다.
    loop = _make_loop(kp_yaw=2.0, yaw_rate_max_deg_s=1000.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=30.0))
    cmd = loop.step((0.0, 0.0, 0.05, 0.0, 0.0, 0.0), time.monotonic())
    # current_grip_deg=0(tcp rz=0), target=30 -> error=30 -> yaw_rate=kp_yaw*30
    assert cmd.yaw_rate == pytest.approx(60.0, abs=1e-6)


def test_yaw_rate_clipped_to_yaw_rate_max_deg_s():
    loop = _make_loop(kp_yaw=10.0, yaw_rate_max_deg_s=15.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=40.0))
    cmd = loop.step((0.0, 0.0, 0.05, 0.0, 0.0, 0.0), time.monotonic())
    # raw = kp_yaw * error = 10 * 40 = 400 -> clip 상한(15)에 걸려야 함
    assert cmd.yaw_rate == pytest.approx(15.0, abs=1e-6)


def test_yaw_valid_false_holds_previous_target():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=50.0))
    assert loop._yaw_target_deg == pytest.approx(50.0)
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))  # yaw_deg=None -> yaw_valid False
    assert loop._yaw_target_deg == pytest.approx(50.0)  # hold, identity로 덮어쓰지 않음


def test_yaw_sign_and_offset_applied_to_current_grip_angle():
    # yaw_offset_deg는 "TCP 로컬 프레임에서 grip_deg=0에 해당하는 기준 벡터의 방향(deg)" -
    # B=0일 때 rot=Rz(C)이므로 기준벡터([cos(offset),sin(offset),0])가 C만큼 더 회전한
    # 뒤 yaw_sign이 곱해진다: current_grip_deg = (yaw_sign * (offset + C)) % 180.
    loop = _make_loop(kp_yaw=1.0, yaw_rate_max_deg_s=1000.0, yaw_sign=-1.0, yaw_offset_deg=10.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=0.0))
    # current_grip_deg = (-1*(10+20)) % 180 = -30 % 180 = 150, target=0 -> 최단경로 오차
    # = ((0-150+90)%180)-90 = 30
    cmd = loop.step((0.0, 0.0, 0.05, 0.0, 0.0, 20.0), time.monotonic())
    assert cmd.yaw_rate == pytest.approx(30.0, abs=1e-6)


def test_current_grip_angle_matches_raw_c_when_b_zero_and_offset_zero():
    # yaw_offset_deg=0(기본값)이면 B=0일 때 회전행렬 기반 계산이 raw C를 그대로 쓰던
    # 이전 구현과 수치적으로 동치여야 한다(Ry(0)=단위행렬) - 회귀 보증.
    loop = _make_loop(kp_yaw=1.0, yaw_rate_max_deg_s=1000.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=50.0))
    cmd = loop.step((0.0, 0.0, 0.05, 0.0, 0.0, 50.0), time.monotonic())
    assert cmd.yaw_rate == pytest.approx(0.0, abs=1e-6)  # 이미 목표(50)와 일치


def test_current_grip_angle_agrees_across_degenerate_zyz_decompositions_at_gimbal_lock():
    # top-down 파지 자세는 B≈180도(ZYZ 짐벌락 특이점) 근방에서 동작한다(named_poses.watch
    # 참고). B=180에서는 ZYZ 오일러 분해가 퇴화한다 - (A,C) 개별 값이 아니라 (A-C)만
    # 물리적 회전을 결정하므로, (A=10,C=20)과 (A=40,C=50)은 서로 다른 raw C(20 vs 50,
    # 30도 차이)를 갖고도 완전히 동일한 물리적 회전이다(수치로 확인:
    # zyz_deg_to_rot(10,180,20) == zyz_deg_to_rot(40,180,50)). raw C를 그대로 "현재
    # 손목 각도"로 쓰면 이 두 경우가 30도나 다른 값으로 읽혀 존재하지도 않는 오차를
    # 좇아 과회전하게 된다(2026-07-13 실기 원인으로 추정) - 회전행렬 투영 방식은 두
    # 경우 모두 같은 current_grip_deg를 내야 한다.
    loop1 = _make_loop(kp_yaw=1.0, yaw_rate_max_deg_s=1000.0)
    loop1.start('spanner', 30.0, 20.0)
    loop1.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=0.0))
    cmd1 = loop1.step((0.0, 0.0, 0.05, 10.0, 180.0, 20.0), time.monotonic())

    loop2 = _make_loop(kp_yaw=1.0, yaw_rate_max_deg_s=1000.0)
    loop2.start('spanner', 30.0, 20.0)
    loop2.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=0.0))
    cmd2 = loop2.step((0.0, 0.0, 0.05, 40.0, 180.0, 50.0), time.monotonic())

    assert cmd1.yaw_rate == pytest.approx(cmd2.yaw_rate, abs=1e-6)


def test_yaw_diverging_triggers_should_abort_when_error_keeps_growing():
    loop = _make_loop(diverge_n_yaw=4, diverge_min_delta_deg=5.0, kp_yaw=0.0,
                       yaw_rate_max_deg_s=1000.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=0.0))
    # kp_yaw=0이라 yaw_rate가 항상 0(로봇이 안 움직이는 시나리오) - tcp rz만 인위적으로
    # 계속 벌려서 |오차|가 단조 증가하는 상황을 재현한다.
    for c in (10.0, 20.0, 30.0, 40.0):
        loop.step((0.0, 0.0, 0.05, 0.0, 0.0, c), time.monotonic())
    assert loop.should_abort() == 'yaw_diverging'


def test_yaw_diverging_does_not_trigger_when_error_converges():
    loop = _make_loop(diverge_n_yaw=4, diverge_min_delta_deg=5.0, kp_yaw=1.0,
                       yaw_rate_max_deg_s=1000.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=30.0))
    # tcp rz가 target(30)에 점점 가까워지는 정상 수렴 상황
    for c in (0.0, 10.0, 20.0, 28.0):
        loop.step((0.0, 0.0, 0.05, 0.0, 0.0, c), time.monotonic())
    assert loop.should_abort() != 'yaw_diverging'


def test_should_close_blocked_until_yaw_settles_then_passes():
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05, n_stable_z=2,
                       cov_threshold=2.5, eps_yaw_deg=2.0, n_stable_yaw=3,
                       kp_yaw=1.0, yaw_rate_max_deg_s=1000.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05, yaw_deg=30.0))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05, yaw_deg=30.0))

    # 손목이 아직 목표(30deg)에서 먼 상태(tcp rz=0) - xy/z/속도 조건은 만족해도 폐합 보류
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0.0, 0.0, 0.0), time.monotonic())
    assert loop.should_close() is False

    # 손목이 목표 각도에 도달(tcp rz=30) - n_stable_yaw주기 연속 후에만 폐합 허용
    cmd = None
    for _ in range(3):
        cmd = loop.step((0.0, 0.0, 0.05, 0.0, 0.0, 30.0), time.monotonic())
    assert cmd.yaw_rate == pytest.approx(0.0, abs=1e-6)
    assert loop.should_close() is True


def test_should_close_ignores_yaw_gate_when_no_yaw_target_ever_observed():
    # 비전 yaw가 아예 실패해도(_yaw_target_deg가 한 번도 안 채워짐) 파지 자체는 막지 않는다.
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05, n_stable_z=2,
                       cov_threshold=2.5, n_stable_yaw=3)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    assert loop._yaw_target_deg is None
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0.0, 0.0, 0.0), time.monotonic())
    assert loop.should_close() is True
