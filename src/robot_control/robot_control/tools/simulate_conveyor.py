"""하드웨어 없이 ServoLoop를 튜닝하기 위한 오프라인 시뮬레이션.
컨베이어 3가지 시나리오(전체 계획.md 7절 4번)를 흉내낸 ToolTrack을 만들어 흘려보내고,
받은 v_cmd를 적분해 tcp_pose를 갱신하며 w/오차를 기록한다."""

import math


class _Stamp:
    def __init__(self, t):
        self.sec = int(t)
        self.nanosec = int((t - int(t)) * 1e9)


class _Header:
    def __init__(self, t):
        self.stamp = _Stamp(t)


class _Position:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Pose:
    def __init__(self, x, y, z):
        self.position = _Position(x, y, z)


class _FakeToolTrack:
    def __init__(self, t, x, y, z, depth_valid=True):
        self.header = _Header(t)
        self.pose = _Pose(x, y, z)
        self.depth_valid = depth_valid


def make_scenario(name, duration_s=5.0, dt=0.02):
    """(t, x, y, z) 리스트. y=0, z=0.05 고정, x만 시나리오에 따라 변화(전부 base_link 기준[m])."""
    n = int(duration_s / dt)
    points = []
    if name == 'constant':
        v = 0.10
        for i in range(n):
            t = i * dt
            points.append((t, 0.9 - v * t, 0.0, 0.05))
    elif name == 'long_reversal':
        v = 0.10
        half = duration_s / 2.0
        for i in range(n):
            t = i * dt
            if t < half:
                x = 0.9 - v * t
            else:
                x = (0.9 - v * half) + v * (t - half)
            points.append((t, x, 0.0, 0.05))
    elif name == 'short_oscillation':
        amplitude = 0.05
        period_s = 0.5
        for i in range(n):
            t = i * dt
            x = 0.7 + amplitude * math.sin(2 * math.pi * t / period_s)
            points.append((t, x, 0.0, 0.05))
    else:
        raise ValueError(f'unknown scenario: {name}')
    return points


def run_servo_sim(loop, scenario, dt=0.02):
    """scenario를 재생하며 매 스텝 ServoLoop를 갱신하고 로그를 반환한다.
    log 각 원소: {'t','w','e_xy','tcp_x','tcp_y'}."""
    tcp_x, tcp_y, tcp_z = scenario[0][1] + 0.3, 0.0, 0.05
    log = []

    loop.start('spanner', 30.0, 20.0)

    for t, x, y, z in scenario:
        loop.on_tool_track(_FakeToolTrack(t, x, y, z))
        cmd = loop.step((tcp_x, tcp_y, tcp_z, 0, 0, 0), t)
        tcp_x += cmd.vx * dt
        tcp_y += cmd.vy * dt
        e_xy = math.hypot(x - tcp_x, y - tcp_y)
        log.append({'t': t, 'w': loop._w, 'e_xy': e_xy, 'tcp_x': tcp_x, 'tcp_y': tcp_y})

    return log


if __name__ == '__main__':
    from robot_control.servo_loop import ServoLoop

    for name in ('constant', 'long_reversal', 'short_oscillation'):
        loop = ServoLoop(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                          eps_descend=0.015, eps_grasp=0.005, n_stable=10,
                          dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                          innov_low=0.010, innov_high=0.040, w_alpha=0.3,
                          diverge_n=15)
        log = run_servo_sim(loop, make_scenario(name, duration_s=4.0, dt=0.02), dt=0.02)
        avg_w = sum(row['w'] for row in log) / len(log)
        avg_e = sum(row['e_xy'] for row in log) / len(log)
        print(f'{name}: avg_w={avg_w:.3f} avg_e_xy={avg_e:.4f}m')
