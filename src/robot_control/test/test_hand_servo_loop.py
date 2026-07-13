import math
import time

import pytest
from robot_control.hand_servo_loop import HandServoLoop, HandServoState


class FakePosition:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakePose:
    def __init__(self, x, y, z):
        self.position = FakePosition(x, y, z)


class FakeHandTrack:
    def __init__(self, x, y, z, detected=True, fist=False):
        self.pose = FakePose(x, y, z)
        self.detected = detected
        self.fist = fist


def _make_loop(**overrides):
    kwargs = dict(kp_xy=1.2, kp_z=1.2, v_max=0.15, offset_m=0.20,
                  t_lost_s=0.3, timeout_s=5.0)
    kwargs.update(overrides)
    return HandServoLoop(**kwargs)


def test_initial_state_is_approaching():
    loop = _make_loop()
    assert loop.get_state() == HandServoState.APPROACHING


def test_step_before_any_track_returns_zero_command():
    loop = _make_loop()
    loop.start()
    cmd = loop.step((0.0, 0.0, 0.0), time.monotonic())
    assert cmd.vx == 0.0 and cmd.vy == 0.0 and cmd.vz == 0.0


def test_step_returns_zero_when_not_detected():
    loop = _make_loop()
    loop.start()
    loop.on_hand_track(FakeHandTrack(1.0, 0.0, 0.0, detected=True))
    loop.on_hand_track(FakeHandTrack(1.0, 0.0, 0.0, detected=False))
    cmd = loop.step((0.0, 0.0, 0.0), time.monotonic())
    assert cmd.vx == 0.0 and cmd.vy == 0.0 and cmd.vz == 0.0


def test_step_moves_toward_offset_point_in_front_of_hand():
    # 손이 TCP 기준 +x 1m, offset_m=0.2 -> 목표는 x=0.8m, 그 방향으로 vx>0이어야 한다.
    loop = _make_loop(offset_m=0.2)
    loop.start()
    loop.on_hand_track(FakeHandTrack(1.0, 0.0, 0.0))
    cmd = loop.step((0.0, 0.0, 0.0), time.monotonic())
    assert cmd.vx > 0.0
    assert cmd.vy == pytest.approx(0.0, abs=1e-9)
    assert cmd.vz == pytest.approx(0.0, abs=1e-9)


def test_step_respects_v_max():
    loop = _make_loop(v_max=0.05, kp_xy=10.0, kp_z=10.0)
    loop.start()
    loop.on_hand_track(FakeHandTrack(1.0, 1.0, 1.0))
    cmd = loop.step((0.0, 0.0, 0.0), time.monotonic())
    assert abs(cmd.vx) <= 0.05 + 1e-9
    assert abs(cmd.vy) <= 0.05 + 1e-9
    assert abs(cmd.vz) <= 0.05 + 1e-9


def test_step_preserves_direction_when_axes_saturate_unevenly():
    # x축 오차가 y/z보다 훨씬 큼 - 축별 독립 클리핑이면 세 축이 다르게(또는
    # 동일하게 v_max로) 눌려 방향이 왜곡된다("커브" 증상의 원인). 벡터 크기
    # 클리핑이면 vx:vy:vz 비율이 원래 오차(e_x:e_y:e_z) 비율과 같아야 한다.
    loop = _make_loop(v_max=0.1, kp_xy=1.0, kp_z=1.0, offset_m=0.0)
    loop.start()
    loop.on_hand_track(FakeHandTrack(10.0, 1.0, 1.0))
    cmd = loop.step((0.0, 0.0, 0.0), time.monotonic())
    raw = (10.0, 1.0, 1.0)
    norm = math.sqrt(sum(c * c for c in raw))
    got_norm = math.sqrt(cmd.vx**2 + cmd.vy**2 + cmd.vz**2)
    assert got_norm == pytest.approx(0.1, abs=1e-9)
    assert cmd.vx / got_norm == pytest.approx(raw[0] / norm, abs=1e-6)
    assert cmd.vy / got_norm == pytest.approx(raw[1] / norm, abs=1e-6)
    assert cmd.vz / got_norm == pytest.approx(raw[2] / norm, abs=1e-6)


def test_step_sets_state_following_when_no_fist():
    loop = _make_loop()
    loop.start()
    loop.on_hand_track(FakeHandTrack(1.0, 0.0, 0.0, fist=False))
    loop.step((0.0, 0.0, 0.0), time.monotonic())
    assert loop.get_state() == HandServoState.FOLLOWING


def test_step_sets_state_stopping_when_fist():
    loop = _make_loop()
    loop.start()
    loop.on_hand_track(FakeHandTrack(1.0, 0.0, 0.0, fist=True))
    loop.step((0.0, 0.0, 0.0), time.monotonic())
    assert loop.get_state() == HandServoState.STOPPING


def test_tick_continue_when_healthy():
    loop = _make_loop()
    loop.start()
    loop.on_hand_track(FakeHandTrack(1.0, 0.0, 0.0))
    status, reason = loop.tick()
    assert status == 'CONTINUE'
    assert reason is None


def test_tick_stop_on_fist():
    loop = _make_loop()
    loop.start()
    loop.on_hand_track(FakeHandTrack(1.0, 0.0, 0.0, fist=True))
    status, reason = loop.tick()
    assert status == 'STOP'
    assert reason == 'fist_detected'


def test_tick_abort_hand_lost():
    loop = _make_loop(t_lost_s=0.0)
    loop.start()
    loop.on_hand_track(FakeHandTrack(1.0, 0.0, 0.0))
    time.sleep(0.01)
    status, reason = loop.tick()
    assert status == 'ABORT'
    assert reason == 'hand_lost'


def test_tick_abort_timeout():
    loop = _make_loop(timeout_s=1.0)
    loop.start()
    loop._start_time = time.monotonic() - 10.0  # 이미 timeout_s를 지난 것처럼 만든다
    status, reason = loop.tick()
    assert status == 'ABORT'
    assert reason == 'timeout'


def test_tick_timeout_disabled_when_non_positive():
    # timeout_s<=0이면 타임아웃 판정 자체가 비활성화된다("주먹까지 계속 추종").
    loop = _make_loop(timeout_s=0.0)
    loop.start()
    loop._start_time = time.monotonic() - 10.0
    status, reason = loop.tick()
    assert status == 'CONTINUE'
    assert reason is None
