"""speedl_stream(비-RT) 서보잉 실현가능성 실측 검증 도구.

RT 세션(ConnectRtControl/StartRtControl) 없이 dsr_msgs2의 speedl_stream 토픽만으로
부드러운 연속 속도 명령이 가능한지, 방향 전환에서도 부드러운지, 발행이 끊겼다
재개될 때 로봇이 어떻게 반응하는지, 이동 중 그리퍼(RG2, Modbus TCP - 완전히 별개
경로)가 정상 동작하는지를 실제 하드웨어에서 사람이 눈으로 확인하기 위한
수동 진단 스크립트다. 자동 성공/실패 판정은 없다 - build_phase_plan()만 순수
함수라 자동 테스트 대상이다.
"""

import argparse
import signal
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
    segments = [PhaseSegment('publish', 'phase1_constant', phase_duration_s, 1)] # 페이즈1, 정방향으로 꾸준히 퍼블리시

    # 페이즈2 오실레이션: osc_period_s 주기로 방향 전환, osc_duration_s 동안 반복
    t = 0.0
    i = 0
    while t < osc_duration_s - 1e-9:
        remaining = osc_duration_s - t
        duration = min(osc_period_s, remaining)
        sign = 1 if i % 2 == 0 else -1
        segments.append(PhaseSegment('publish', f'phase2_osc_{i}', duration, sign))
        t += duration
        i += 1

    # 페이즈3
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
    # dsr_controller2가 토픽 이름 앞에 자기 노드 이름을 붙이는지는 doosan-robot2
    # 릴리스에 따라 다르다(robot_control_node의 doosan_driver.controller_name과
    # 동일한 이유). 기본값은 팀 대부분이 쓰는 옛 버전(세그먼트 없음) 기준이므로
    # 2026-03-06 이후의 새 포크를 쓰면 --controller-name dsr_controller2로 넘긴다.
    parser.add_argument('--controller-name', default='')
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
            print('정지 명령 1회 발행 후 침묵 - 로봇이 멈춘 채 유지되는지 관찰하세요.')
            _publish_zero(pub, speedl_stream_cls, acc, period_s, repeats=1)
            print(f'발행 중단 {segment.duration_s:.2f}s - 로봇 반응을 관찰하세요.')
            ok = _sleep_ticked(stop_event, segment.duration_s)
        if not ok:
            return False
    return True


def _run_gripper_during_motion(
        pub, speedl_stream_cls, axis, vel_mm_s, acc, period_s,
        rg2_client, grasp_width_mm, grasp_force_n, stop_event):
    """메인 스레드가 speedl_stream을 계속 발행하는 동안, 별도 스레드에서 그리퍼를
    닫았다 여는 것으로 '이동 중 그리퍼 동작'을 확인한다. RG2Client.close/open은
    내부적으로 busy 비트를 폴링하며 블로킹하므로 메인 발행 루프와 분리해야 한다."""
    gripper_done = threading.Event()

    def _gripper_worker():
        print('[그리퍼] close() 호출')
        ok_close = rg2_client.close(grasp_width_mm, grasp_force_n)
        print(f'[그리퍼] close 결과: {ok_close} (status={rg2_client.last_status})')
        time.sleep(1.0)
        print('[그리퍼] open() 호출')
        ok_open = rg2_client.open()
        print(f'[그리퍼] open 결과: {ok_open} (status={rg2_client.last_status})')
        gripper_done.set()

    worker = threading.Thread(target=_gripper_worker, daemon=True)
    worker.start()

    msg = speedl_stream_cls()
    msg.vel = _velocity_vector(axis, vel_mm_s, 1)
    msg.acc = list(acc)
    msg.time = period_s
    print(f'--- phase5_gripper_during_motion ({axis}축 발행 지속, 그리퍼 스레드 종료 대기) ---')
    print('주의: 로봇이 다시 움직입니다. 그리퍼가 끝날 때까지(최대 ~10초) 계속 이동합니다.')
    while not gripper_done.is_set():
        if stop_event.is_set():
            break
        pub.publish(msg)
        time.sleep(period_s)
    worker.join(timeout=5.0)


def main(argv=None):
    args = _parse_args(argv)
    _confirm_or_exit()

    try:
        import rclpy
        from dsr_msgs2.msg import SpeedlStream
        from robot_control.rg2_client import RG2Client
    except ImportError as exc:
        print(f'dsr_msgs2/rclpy import 실패 - 두산 ROS2 워크스페이스를 source 하세요: {exc}')
        sys.exit(1)

    acc = (args.acc_trans_mm_s2, args.acc_rot_deg_s2)
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        print(f'\n시그널 {signum} 수신 - 정지 명령 발행 후 종료합니다.')
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    rclpy.init(args=None)
    node = rclpy.create_node('probe_speedl_stream')
    prefix = (f'/{args.robot_id}/{args.controller_name}'
              if args.controller_name else f'/{args.robot_id}')
    pub = node.create_publisher(SpeedlStream, f'{prefix}/speedl_stream', 10)
    rg2_client = RG2Client(
        ip=args.rg2_ip, port=args.rg2_port, hardware_enabled=True,
        gripper=args.rg2_gripper, node=None)

    try:
        segments = build_phase_plan(
            phase_duration_s=args.phase_duration_s,
            osc_period_s=args.osc_period_s,
            osc_duration_s=args.osc_duration_s,
            pause_durations_s=args.pause_durations_s)
        completed = _run_phase_segments(
            pub, SpeedlStream, args.axis, args.vel_mm_s, acc,
            args.period_s, segments, stop_event)

        print('--- phase4_explicit_stop ---')
        _publish_zero(pub, SpeedlStream, acc, args.period_s) # 정지

        if completed and not stop_event.is_set():
            _run_gripper_during_motion( # 이동 중 그리퍼 동작 확인
                pub, SpeedlStream, args.axis, args.vel_mm_s, acc,
                args.period_s, rg2_client, args.grasp_width_mm,
                args.grasp_force_n, stop_event)

        print('--- phase6_final_stop ---')
        _publish_zero(pub, SpeedlStream, acc, args.period_s) # 최종 정지
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
