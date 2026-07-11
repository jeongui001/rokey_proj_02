"""수집된 ToolTrack CSV(tools/tool_track_logger.py 출력)를 KalmanXYZV/
ServoLoop(운영 코드)에 그대로 재생해 파라미터 후보를 비교하는 오프라인
튜닝 도구. rclpy 불필요 - 순수 Python.

docs/superpowers/specs/2026-07-11-kalman-servo-control-tuning-design.md
1~5단계 참고.
"""

import argparse
import csv


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
