import pytest
from robot_control.servo_loop import ServoLoop
from robot_control.tools.simulate_conveyor import make_scenario, run_servo_sim, _parse_args


def _make_loop():
    return ServoLoop(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                      eps_descend=0.015, eps_grasp=0.005, n_stable=5,
                      dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                      innov_low=0.010, innov_high=0.040, w_alpha=0.3)


def test_constant_scenario_keeps_w_near_one():
    loop = _make_loop()
    scenario = make_scenario('constant', duration_s=3.0, dt=0.02)
    log = run_servo_sim(loop, scenario, dt=0.02)
    tail_w = [row['w'] for row in log[-20:]]
    assert sum(tail_w) / len(tail_w) > 0.8


def test_long_reversal_scenario_drops_w_after_direction_change_then_recovers():
    # 0.10m/s 벨트를 30~60Hz로 관측하면 방향전환 순간의 프레임당 잔차는 몇 mm 수준이라
    # innov_high(40mm)를 단번에 넘기지는 않는다 — 실제 물리 벨트도 순간 반전이 아니라
    # 감속 후 반대로 가속하므로, 여기서는 "정상 구간보다 눈에 띄게 낮아졌다가 회복"만 확인한다.
    loop = _make_loop()
    scenario = make_scenario('long_reversal', duration_s=4.0, dt=0.02)
    log = run_servo_sim(loop, scenario, dt=0.02)
    mid = len(log) // 2
    steady_w = log[mid - 10]['w']
    min_w_after_reversal = min(row['w'] for row in log[mid:mid + 25])
    recovered_w = log[-1]['w']
    assert min_w_after_reversal < steady_w - 0.01
    assert recovered_w > min_w_after_reversal


def test_short_oscillation_scenario_produces_lower_average_w_than_constant():
    loop_const = _make_loop()
    loop_osc = _make_loop()
    const_log = run_servo_sim(loop_const, make_scenario('constant', 3.0, 0.02), dt=0.02)
    osc_log = run_servo_sim(loop_osc, make_scenario('short_oscillation', 3.0, 0.02), dt=0.02)

    avg_w_const = sum(row['w'] for row in const_log) / len(const_log)
    avg_w_osc = sum(row['w'] for row in osc_log) / len(osc_log)
    assert avg_w_osc < avg_w_const


def test_parse_args_defaults_to_current_yaml_kp_xy():
    args = _parse_args([])
    assert args.kp_xy == [1.2]


def test_parse_args_accepts_multiple_kp_xy_candidates():
    args = _parse_args(['--kp-xy', '1.0', '1.5', '2.0'])
    assert args.kp_xy == [1.0, 1.5, 2.0]
