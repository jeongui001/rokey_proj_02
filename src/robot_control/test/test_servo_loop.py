import time

import pytest
from geometry_msgs.msg import PoseStamped
from handover_interfaces.msg import ToolTrack
from robot_control.servo_loop import HandApproachServo, HandApproachState, ServoLoop, ServoState


def _make_loop(**overrides):
    kwargs = dict(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                  eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                  dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3)
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


# ---- 좌표 계약: position은 (target - current) 오차. v = +Kp*error로 오차를 줄인다 ----

def test_step_commands_velocity_toward_positive_error():
    # target이 TCP의 +x/+y 방향에 있으면(error>0), 그 방향으로 움직이는 명령(+)을 낸다.
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=0.1, y=0.05, z=0.05))

    cmd = loop.step()

    assert cmd.vx > 0.0  # +x 오차 -> +x로 이동
    assert cmd.vy > 0.0  # +y 오차 -> +y로 이동
    assert cmd.vz == 0.0  # xy 오차가 커서 아직 z축은 움직이지 않음


def test_step_commands_velocity_toward_negative_error():
    # target이 TCP의 -x/-y 방향에 있으면(error<0), 그 방향(-)으로 이동한다.
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=-0.1, y=-0.05, z=0.05))

    cmd = loop.step()

    assert cmd.vx < 0.0
    assert cmd.vy < 0.0


def test_step_clips_velocity_to_v_max():
    loop = _make_loop(v_max=0.2)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=10.0, y=0.0, z=0.05))

    cmd = loop.step()

    assert cmd.vx == pytest.approx(0.2)  # +방향 오차 -> +로 clip


def test_step_approaches_downward_when_object_below_tcp():
    # 물체가 TCP보다 아래에 있으면(error_z<0, base_link 프레임 기준) z 속도는 음수(-)여야 한다.
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=0.001, y=0.001, z=-0.05))

    cmd = loop.step()

    assert cmd.vz < 0.0


def test_step_approaches_upward_when_object_above_tcp():
    # 반대로 물체가 TCP보다 위에 있으면(error_z>0) z 속도는 양수(+)여야 한다.
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=0.001, y=0.001, z=0.05))

    cmd = loop.step()

    assert cmd.vz > 0.0


def test_step_z_velocity_clipped_to_descend_speed():
    loop = _make_loop(descend_speed=0.05)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(_track(x=0.001, y=0.001, z=-10.0))

    cmd = loop.step()

    assert cmd.vz == pytest.approx(-0.05)


def test_step_yaw_locked_to_zero_by_default_regardless_of_orientation():
    # enable_yaw_control 기본값(false) - 절대 orientation을 그대로 yaw 명령으로 쓰지 않는다.
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    msg = _track(x=0.001, y=0.001, z=0.0)
    msg.pose.orientation.z = 0.7071
    msg.pose.orientation.w = 0.7071  # 90도 회전 - 무시되어야 한다
    loop.on_tool_track(msg)

    cmd = loop.step()

    assert cmd.yaw_rate == 0.0


def test_step_yaw_control_enabled_uses_orientation():
    loop = _make_loop(enable_yaw_control=True)
    loop.start('spanner', 30.0, 20.0)
    msg = _track(x=0.001, y=0.001, z=0.0)
    msg.pose.orientation.z = 0.7071
    msg.pose.orientation.w = 0.7071
    loop.on_tool_track(msg)

    cmd = loop.step()

    assert cmd.yaw_rate != 0.0


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


def test_should_close_uses_absolute_z_error_when_negative():
    # error_z가 음수(물체가 TCP 아래)여도 절댓값이 z_close_m 이내면 close로 판단한다.
    loop = _make_loop(n_stable=1, eps_grasp=0.005, z_close_m=0.01)
    loop.start('spanner', 30.0, 20.0)

    loop.on_tool_track(_track(x=0.001, y=0.001, z=-0.005))

    assert loop.should_close() is True


def test_should_close_false_if_negative_z_gap_still_large():
    loop = _make_loop(n_stable=1, eps_grasp=0.005, z_close_m=0.01)
    loop.start('spanner', 30.0, 20.0)

    loop.on_tool_track(_track(x=0.001, y=0.001, z=-0.05))

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
