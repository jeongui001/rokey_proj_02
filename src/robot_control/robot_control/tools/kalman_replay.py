"""수집된 ToolTrack CSV(tools/tool_track_logger.py 출력)를 KalmanXYZV/
ServoLoop(운영 코드)에 그대로 재생해 파라미터 후보를 비교하는 오프라인
튜닝 도구. rclpy 불필요 - 순수 Python.

docs/superpowers/specs/2026-07-11-kalman-servo-control-tuning-design.md
1~5단계 참고.
"""

import argparse
import csv
from unittest import mock

from robot_control.kalman import KalmanXYZV
from robot_control.servo_loop import ServoLoop


class TrackRow:
    def __init__(self, stamp_s, recv_monotonic_s, x, y, z, depth_valid):
        self.stamp_s = stamp_s
        self.recv_monotonic_s = recv_monotonic_s
        self.x = x
        self.y = y
        self.z = z
        self.depth_valid = depth_valid


def _to_bool(value):
    return value.strip().lower() in ('true', '1')


def read_track_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(TrackRow(
                stamp_s=float(r['stamp_s']),
                recv_monotonic_s=float(r['recv_monotonic_s']),
                x=float(r['x']), y=float(r['y']), z=float(r['z']),
                depth_valid=_to_bool(r['depth_valid'])))
    return rows


def write_replay_csv(path, records):
    if not records:
        raise ValueError('records가 비어 있습니다 - 재생 결과가 없습니다.')
    fieldnames = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def replay_kalman(rows, q_pos, q_vel, r_xy, r_z, p0_vel_reset):
    """rows를 KalmanXYZV.predict()/update_*()에 순서대로 주입 - ServoLoop의
    w/innovation 스무딩 없이 필터 자체의 수렴만 본다(1단계 r_xy/r_z 검증용)."""
    kf = KalmanXYZV(q_pos=q_pos, q_vel=q_vel, r_xy=r_xy, r_z=r_z, p0_vel_reset=p0_vel_reset)
    records = []
    prev_stamp = None
    for row in rows:
        if not kf._initialized:
            kf.initialize(row.x, row.y, row.z)
            prev_stamp = row.stamp_s
            innovation = None
        else:
            dt = max(row.stamp_s - prev_stamp, 1e-3)
            kf.predict(dt)
            if row.depth_valid:
                innovation = kf.update_xyz([row.x, row.y, row.z])
            else:
                innovation = kf.update_xy_only([row.x, row.y])
            prev_stamp = row.stamp_s
        records.append({
            'stamp_s': row.stamp_s,
            'innovation_xy_m': innovation,
            'x': kf.position[0], 'y': kf.position[1], 'z': kf.position[2],
            'vx': kf.velocity[0], 'vy': kf.velocity[1],
        })
    return records


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


class _ReplayToolTrack:
    def __init__(self, row):
        self.header = _Header(row.stamp_s)
        self.pose = _Pose(row.x, row.y, row.z)
        self.depth_valid = row.depth_valid


# ServoLoop 생성자가 요구하지만 이 도구의 튜닝 대상이 아닌 파라미터 - 현재
# robot_control_params.yaml의 servo: 섹션 기본값과 동일하게 고정한다.
_SERVO_FIXED_KWARGS = dict(
    kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.05,
    eps_descend=0.015, eps_grasp=1.0, n_stable=1, timeout_s=10.0,
    z_close=0.03, diverge_n=15, cov_threshold=100.0)


def replay_servo(rows, dt_latency, t_lost_s, innov_low, innov_high, w_alpha,
                  q_pos, q_vel, r_xy, r_z, p0_vel_reset):
    """rows를 ServoLoop.on_tool_track()에 순서대로 주입하며, 매 행 주입 직전에
    should_abort()를 호출해 그 행 도착 전까지의 경과시간 기준으로 t_lost/timeout
    오탐 여부를 기록한다(3단계). step()은 실측 TCP가 없어 호출하지 않으므로
    e_xy_norm/z_gap/should_close/'diverging'은 이 함수로 검증할 수 없다.
    ServoLoop 내부가 time.monotonic()을 직접 호출하므로, 로그의
    recv_monotonic_s를 그대로 '현재 시각'으로 쓰도록 monkeypatch한다."""
    loop = ServoLoop(
        dt_latency=dt_latency, t_lost_s=t_lost_s, innov_low=innov_low,
        innov_high=innov_high, w_alpha=w_alpha, q_pos=q_pos, q_vel=q_vel,
        r_xy=r_xy, r_z=r_z, p0_vel_reset=p0_vel_reset, **_SERVO_FIXED_KWARGS)

    clock = {'t': rows[0].recv_monotonic_s}

    def _fake_monotonic():
        return clock['t']

    records = []
    with mock.patch('robot_control.servo_loop.time.monotonic', side_effect=_fake_monotonic):
        loop.start('spanner', 30.0, 20.0)
        for row in rows:
            clock['t'] = row.recv_monotonic_s
            abort_reason = loop.should_abort()
            loop.on_tool_track(_ReplayToolTrack(row))
            snap = loop.debug_snapshot()
            records.append({
                'stamp_s': row.stamp_s,
                'w': snap['w'],
                'w_target': snap['w_target'],
                'innovation_xy_m': snap['innovation_xy_m'],
                'depth_valid': snap['depth_valid'],
                'abort_reason': abort_reason or '',
            })
    return records
