"""수집된 ToolTrack CSV(tools/tool_track_logger.py 출력)를 KalmanXYZV/
ServoLoop(운영 코드)에 그대로 재생해 파라미터 후보를 비교하는 오프라인
튜닝 도구. rclpy 불필요 - 순수 Python.

docs/superpowers/specs/2026-07-11-kalman-servo-control-tuning-design.md
1~5단계 참고.
"""

import argparse
import csv

from robot_control.kalman import KalmanXYZV


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
