"""speedl_stream(비-RT) 서보잉 실현가능성 실측 검증 도구.

RT 세션(ConnectRtControl/StartRtControl) 없이 dsr_msgs2의 speedl_stream 토픽만으로
부드러운 연속 속도 명령이 가능한지, 방향 전환에서도 부드러운지, 발행이 끊겼다
재개될 때 로봇이 어떻게 반응하는지, 이동 중 그리퍼(RG2, Modbus TCP - 완전히 별개
경로)가 정상 동작하는지를 실제 하드웨어에서 사람이 눈으로 확인하기 위한
수동 진단 스크립트다. 자동 성공/실패 판정은 없다 - build_phase_plan()만 순수
함수라 자동 테스트 대상이다.
"""

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
