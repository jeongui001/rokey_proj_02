"""speedl_stream(비-RT) 서보잉 실현가능성 실측 검증 도구.

RT 세션(ConnectRtControl/StartRtControl) 없이 dsr_msgs2의 speedl_stream 토픽만으로
부드러운 연속 속도 명령이 가능한지, 방향 전환에서도 부드러운지, 발행이 끊겼다
재개될 때 로봇이 어떻게 반응하는지, 이동 중 그리퍼(RG2, Modbus TCP - 완전히 별개
경로)가 정상 동작하는지를 실제 하드웨어에서 사람이 눈으로 확인하기 위한
수동 진단 스크립트다. 자동 성공/실패 판정은 없다 - build_phase_plan()만 순수
함수라 자동 테스트 대상이다.
"""

import argparse
import sys
import threading
import time
from collections import namedtuple

AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}

PhaseSegment = namedtuple('PhaseSegment', ['kind', 'label', 'duration_s', 'sign'])


def build_phase_plan(
        phase_duration_s, osc_period_s, osc_duration_s,
        pause_durations_s, pause_burst_s=1.0):
    """1~3단계(일정속도 지속발행 / 방향전환 오시레이션 / 명령 중단-재개) 스케줄을
    만든다. axis/vel_mm_s는 세그먼트에 담지 않는다 - sign(부호)만 담고, 실제 축과
    속도 크기는 실행 시점에 적용한다(_velocity_vector 참고)."""
    segments = [PhaseSegment('publish', 'phase1_constant', phase_duration_s, 1)]

    t = 0.0
    i = 0
    while t < osc_duration_s - 1e-9:
        remaining = osc_duration_s - t
        duration = min(osc_period_s, remaining)
        sign = 1 if i % 2 == 0 else -1
        segments.append(PhaseSegment('publish', f'phase2_osc_{i}', duration, sign))
        t += duration
        i += 1

    for i, pause_s in enumerate(pause_durations_s):
        segments.append(PhaseSegment('publish', f'phase3_burst_{i}', pause_burst_s, 1))
        segments.append(PhaseSegment('pause', f'phase3_pause_{i}', pause_s, 0))

    return segments


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'speedl_stream(비-RT) 서보잉 실현가능성 실측 검증 도구 - '
            '실제 로봇이 움직입니다.'))
    parser.add_argument('--robot-id', default='dsr01')
    parser.add_argument('--axis', default='x', choices=list(AXIS_INDEX))
    parser.add_argument('--vel-mm-s', type=float, default=20.0)
    parser.add_argument('--acc-trans-mm-s2', type=float, default=100.0)
    parser.add_argument('--acc-rot-deg-s2', type=float, default=100.0)
    parser.add_argument('--phase-duration-s', type=float, default=3.0)
    parser.add_argument('--osc-period-s', type=float, default=1.0)
    parser.add_argument('--osc-duration-s', type=float, default=4.0)
    parser.add_argument(
        '--pause-durations-s', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    parser.add_argument('--period-s', type=float, default=0.02)
    parser.add_argument('--rg2-ip', default='192.168.1.1')
    parser.add_argument('--rg2-port', type=int, default=502)
    parser.add_argument('--rg2-gripper', default='rg2')
    parser.add_argument('--grasp-width-mm', type=float, required=True)
    parser.add_argument('--grasp-force-n', type=float, required=True)
    return parser.parse_args(argv)


def _confirm_or_exit():
    print('=' * 70)
    print('경고: 이 스크립트는 실제 로봇을 speedl_stream(비-RT)으로 움직입니다.')
    print('Enable 스위치/펜던트를 손에 쥔 상태에서만 계속하세요.')
    print('단위 가정(mm/s, deg/s)이 틀렸을 가능성이 있으니 1단계 시작 직후')
    print('몇 초는 특히 주의 깊게 관찰하세요.')
    print('=' * 70)
    answer = input("계속하려면 'yes'를 입력하세요: ").strip().lower()
    if answer != 'yes':
        print('취소되었습니다.')
        sys.exit(0)


STOP_TICK_S = 0.05
FINAL_STOP_REPEATS = 5


def _velocity_vector(axis, vel_mm_s, sign):
    vel6 = [0.0] * 6
    vel6[AXIS_INDEX[axis]] = vel_mm_s * sign
    return vel6


def _sleep_ticked(stop_event, seconds):
    """seconds초를 STOP_TICK_S 단위로 나눠 자며 stop_event를 감시한다.
    stop_event가 세팅되면 즉시 False, 다 자면 True를 반환한다."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_event.is_set():
            return False
        time.sleep(min(STOP_TICK_S, max(deadline - time.monotonic(), 0.0)))
    return True


def _publish_zero(pub, speedl_stream_cls, acc, period_s, repeats=FINAL_STOP_REPEATS):
    msg = speedl_stream_cls()
    msg.vel = [0.0] * 6
    msg.acc = list(acc)
    msg.time = period_s
    for _ in range(repeats):
        pub.publish(msg)
        time.sleep(period_s)


def _run_publish_segment(
        pub, speedl_stream_cls, axis, vel_mm_s, acc, period_s, segment, stop_event):
    msg = speedl_stream_cls()
    msg.vel = _velocity_vector(axis, vel_mm_s, segment.sign)
    msg.acc = list(acc)
    msg.time = period_s
    deadline = time.monotonic() + segment.duration_s
    while time.monotonic() < deadline:
        if stop_event.is_set():
            return False
        pub.publish(msg)
        time.sleep(period_s)
    return True


def _run_phase_segments(
        pub, speedl_stream_cls, axis, vel_mm_s, acc, period_s, segments, stop_event):
    for segment in segments:
        print(f'--- {segment.label} ({segment.kind}, {segment.duration_s:.2f}s) ---')
        if segment.kind == 'publish':
            ok = _run_publish_segment(
                pub, speedl_stream_cls, axis, vel_mm_s, acc, period_s, segment,
                stop_event)
        else:
            print(f'발행 중단 {segment.duration_s:.2f}s - 로봇 반응을 관찰하세요.')
            ok = _sleep_ticked(stop_event, segment.duration_s)
        if not ok:
            return False
    return True
