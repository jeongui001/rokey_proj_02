"""칼만필터 튜닝용 /vision/tool_track 로그 CSV 저장 도구.

수집한 CSV는 tools/kalman_replay.py의 입력으로 쓰여 KalmanXYZV/ServoLoop
파라미터를 실기 없이 오프라인 재생으로 튜닝하는 데 쓰인다
(docs/superpowers/specs/2026-07-11-kalman-servo-control-tuning-design.md
1/3/4/5단계 참고). probe_speedl_stream.py와 동일하게 자동 성공/실패 판정은
없는 수동 데이터 수집 도구다.
"""

import argparse
import csv
import signal
import sys
import threading

CSV_HEADER = ['stamp_s', 'recv_monotonic_s', 'x', 'y', 'z', 'depth_valid']


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='/vision/tool_track을 CSV로 기록 - kalman_replay.py 입력용.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--topic', default='/vision/tool_track')
    return parser.parse_args(argv)


def format_row(msg, recv_monotonic_s):
    """ToolTrack 메시지 1개를 CSV 행(dict)으로 변환. rclpy 의존 없음."""
    stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return {
        'stamp_s': stamp_s,
        'recv_monotonic_s': recv_monotonic_s,
        'x': msg.pose.position.x,
        'y': msg.pose.position.y,
        'z': msg.pose.position.z,
        'depth_valid': bool(msg.depth_valid),
    }


def main(argv=None):
    args = _parse_args(argv)

    try:
        import time

        import rclpy
        from handover_interfaces.msg import ToolTrack
    except ImportError as exc:
        print(f'rclpy/handover_interfaces import 실패 - ROS2 워크스페이스를 source 하세요: {exc}')
        sys.exit(1)

    out_file = open(args.out, 'w', newline='')
    writer = csv.DictWriter(out_file, fieldnames=CSV_HEADER)
    writer.writeheader()
    count = 0
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    rclpy.init(args=None)
    node = rclpy.create_node('tool_track_logger')

    def _on_tool_track(msg):
        nonlocal count
        writer.writerow(format_row(msg, time.monotonic()))
        out_file.flush()
        count += 1

    node.create_subscription(ToolTrack, args.topic, _on_tool_track, 10)

    print(f'{args.topic} 구독 시작 - Ctrl+C로 종료, 저장 경로: {args.out}')
    try:
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        out_file.close()
        print(f'{count}개 프레임 저장 완료: {args.out}')


if __name__ == '__main__':
    main()
