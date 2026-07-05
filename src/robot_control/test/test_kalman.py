import pytest
from robot_control.kalman import KalmanXYZV


def test_initialize_sets_position_and_zero_velocity():
    kf = KalmanXYZV()
    kf.initialize(1.0, 2.0, 0.05)
    assert list(kf.position) == pytest.approx([1.0, 2.0, 0.05])
    assert list(kf.velocity) == pytest.approx([0.0, 0.0])
    assert kf._initialized is True


def test_predict_advances_position_by_velocity():
    kf = KalmanXYZV()
    kf.initialize(0.0, 0.0, 0.05)
    kf.x[3] = 0.1
    kf.predict(dt=1.0)
    assert kf.position[0] == pytest.approx(0.1, abs=1e-6)


def test_update_xyz_returns_innovation_and_pulls_toward_measurement():
    kf = KalmanXYZV(q_pos=1e-3, q_vel=1e-2, r_xy=1e-4, r_z=1e-4)
    kf.initialize(0.0, 0.0, 0.05)
    kf.predict(dt=0.02)
    innov = kf.update_xyz([0.002, 0.0, 0.05])
    assert innov == pytest.approx(0.002, abs=1e-6)
    assert 0.0 < kf.position[0] < 0.002


def test_update_xy_only_leaves_z_unchanged():
    kf = KalmanXYZV()
    kf.initialize(0.0, 0.0, 0.05)
    kf.predict(dt=0.02)
    kf.update_xy_only([0.001, 0.0])
    assert kf.position[2] == pytest.approx(0.05, abs=1e-9)


def test_reset_velocity_covariance_sets_trace_to_2x_p0():
    kf = KalmanXYZV(p0_vel_reset=2.0)
    kf.initialize(0.0, 0.0, 0.05)
    kf.predict(dt=0.02)
    kf.update_xyz([0.0, 0.0, 0.05])
    kf.reset_velocity_covariance()
    assert kf.velocity_covariance_trace == pytest.approx(4.0)


def test_predict_position_does_not_mutate_state():
    kf = KalmanXYZV()
    kf.initialize(0.0, 0.0, 0.05)
    kf.x[3] = 0.1
    before = kf.position.copy()
    p = kf.predict_position(0.5)
    assert list(kf.position) == pytest.approx(list(before))
    assert p[0] == pytest.approx(0.05)
    assert p[2] == pytest.approx(0.05)
