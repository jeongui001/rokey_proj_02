import pytest
from robot_control.servo_loop import ServoLoop, ServoState


def _make_loop():
    return ServoLoop(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                      eps_descend=0.015, eps_grasp=0.005, n_stable=5,
                      dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3)


def test_initial_state_is_tracking():
    loop = _make_loop()
    assert loop.get_state() == ServoState.TRACKING


def test_stub_methods_raise_not_implemented():
    loop = _make_loop()
    with pytest.raises(NotImplementedError):
        loop.start('spanner', 30.0, 20.0)
    with pytest.raises(NotImplementedError):
        loop.on_tool_track(object())
    with pytest.raises(NotImplementedError):
        loop.step()
    with pytest.raises(NotImplementedError):
        loop.should_close()
    with pytest.raises(NotImplementedError):
        loop.should_abort()
