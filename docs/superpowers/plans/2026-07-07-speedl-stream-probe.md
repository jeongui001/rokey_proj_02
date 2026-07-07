# speedl_stream 비-RT 서보잉 검증 스크립트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RT 세션(`ConnectRtControl`/`StartRtControl`) 없이 `dsr_msgs2`의 `speedl_stream` 토픽만으로 부드러운 연속 속도 서보잉이 가능한지 실제 로봇에서 검증하는 수동 진단 스크립트를 만든다.

**Architecture:** `robot_control` 패키지의 `tools/` 아래에 단일 파일 스크립트를 추가한다. 순수 로직(속도 스케줄 생성)과 하드웨어 I/O(퍼블리시, 그리퍼 호출)를 함수 단위로 분리해 앞부분만 자동 테스트하고, 나머지는 사람이 실제 로봇을 보고 판단하는 수동 진단 흐름으로 둔다. RT 세션 코드(`doosan_driver.py`)는 건드리지 않는다 — 완전히 별도 경로다.

**Tech Stack:** rclpy, dsr_msgs2 (SpeedlStream), 기존 `robot_control.rg2_client.RG2Client`, argparse, threading, pytest.

## Global Constraints

- 기본값은 매우 보수적이어야 한다: 선속도 20mm/s, 가속 100mm/s²(병진)/100deg/s²(회전), 각 구간 3~5초.
- `dsr_msgs2`/`rclpy` import는 `doosan_driver.py`와 동일하게 함수 내부 지연 import로 처리한다 (모듈 최상단에서 하지 않는다).
- `--grasp-width-mm`/`--grasp-force-n`은 기본값 없이 필수 인자로 요구한다 (하드웨어별 실측값을 추측하지 않는다).
- 실행 전 반드시 인터랙티브 확인("yes" 입력)을 받은 뒤에만 퍼블리시를 시작한다.
- SIGINT/SIGTERM 수신 시 `vel=[0]*6`을 여러 번 발행한 뒤 종료한다.
- 테스트는 `build_phase_plan` 순수 함수만 대상으로 한다 (스크립트 전체는 실제 로봇 관찰이 성공 기준이라 자동 테스트 대상이 아님 — 스펙에 명시됨).
- 테스트 실행 명령: `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd)" python3 -m pytest test/test_probe_speedl_stream.py -v` (이 환경의 pytest는 `anyio` 플러그인과 충돌하므로 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`이 필요함 — 기존 `test_kalman.py`로 확인됨).

---

### Task 1: `build_phase_plan` 순수 함수 (속도 스케줄 생성)

**Files:**
- Create: `src/robot_control/robot_control/tools/probe_speedl_stream.py`
- Test: `src/robot_control/test/test_probe_speedl_stream.py`

**Interfaces:**
- Produces: `PhaseSegment` namedtuple(`kind: str`(`'publish'`|`'pause'`), `label: str`, `duration_s: float`, `sign: int`); `build_phase_plan(phase_duration_s, osc_period_s, osc_duration_s, pause_durations_s, pause_burst_s=1.0) -> list[PhaseSegment]`.

- [ ] **Step 1: 실패하는 테스트 작성**

`src/robot_control/test/test_probe_speedl_stream.py` 생성:

```python
from robot_control.tools.probe_speedl_stream import PhaseSegment, build_phase_plan


def test_phase1_constant_velocity_segment():
    segments = build_phase_plan(
        phase_duration_s=3.0, osc_period_s=1.0, osc_duration_s=0.0,
        pause_durations_s=[])
    assert segments[0] == PhaseSegment('publish', 'phase1_constant', 3.0, 1)


def test_oscillation_alternates_sign_each_period():
    segments = build_phase_plan(
        phase_duration_s=0.0, osc_period_s=1.0, osc_duration_s=4.0,
        pause_durations_s=[])
    osc_segments = [s for s in segments if s.label.startswith('phase2_osc_')]
    assert [s.sign for s in osc_segments] == [1, -1, 1, -1]
    assert all(s.duration_s == 1.0 for s in osc_segments)


def test_oscillation_last_segment_truncated_to_remaining_duration():
    segments = build_phase_plan(
        phase_duration_s=0.0, osc_period_s=1.0, osc_duration_s=2.5,
        pause_durations_s=[])
    osc_segments = [s for s in segments if s.label.startswith('phase2_osc_')]
    assert [s.duration_s for s in osc_segments] == [1.0, 1.0, 0.5]
    assert [s.sign for s in osc_segments] == [1, -1, 1]


def test_pause_resume_segments_preserve_order_and_alternate_kind():
    segments = build_phase_plan(
        phase_duration_s=0.0, osc_period_s=1.0, osc_duration_s=0.0,
        pause_durations_s=[0.5, 1.0, 2.0], pause_burst_s=1.0)
    phase3 = [s for s in segments if s.label.startswith('phase3_')]
    assert [s.kind for s in phase3] == [
        'publish', 'pause', 'publish', 'pause', 'publish', 'pause']
    assert [s.duration_s for s in phase3 if s.kind == 'pause'] == [0.5, 1.0, 2.0]
    assert all(s.duration_s == 1.0 for s in phase3 if s.kind == 'publish')


def test_full_plan_concatenates_all_three_phases_in_order():
    segments = build_phase_plan(
        phase_duration_s=3.0, osc_period_s=1.0, osc_duration_s=2.0,
        pause_durations_s=[0.5, 1.0])
    labels = [s.label for s in segments]
    assert labels == [
        'phase1_constant',
        'phase2_osc_0', 'phase2_osc_1',
        'phase3_burst_0', 'phase3_pause_0',
        'phase3_burst_1', 'phase3_pause_1',
    ]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd)" python3 -m pytest test/test_probe_speedl_stream.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robot_control.tools.probe_speedl_stream'`

- [ ] **Step 3: 최소 구현 작성**

`src/robot_control/robot_control/tools/probe_speedl_stream.py` 생성:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd)" python3 -m pytest test/test_probe_speedl_stream.py -v`
Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/robot_control/robot_control/tools/probe_speedl_stream.py src/robot_control/test/test_probe_speedl_stream.py
git commit -m "feat(robot_control): add build_phase_plan for speedl_stream probe script"
```

---

### Task 2: CLI 인자 파싱 + 실행 전 확인 프롬프트

**Files:**
- Modify: `src/robot_control/robot_control/tools/probe_speedl_stream.py`

**Interfaces:**
- Consumes: `AXIS_INDEX`(Task 1)
- Produces: `_parse_args(argv=None) -> argparse.Namespace`(필드: `robot_id, axis, vel_mm_s, acc_trans_mm_s2, acc_rot_deg_s2, phase_duration_s, osc_period_s, osc_duration_s, pause_durations_s, period_s, rg2_ip, rg2_port, rg2_gripper, grasp_width_mm, grasp_force_n`); `_confirm_or_exit() -> None`

이 함수들은 스펙에 따라 자동 테스트 대상이 아니다(인터랙티브 입력/프로세스 종료). Step 2에서 `--help`로 수동 스모크 검증한다.

- [ ] **Step 1: `_parse_args`/`_confirm_or_exit` 추가**

`src/robot_control/robot_control/tools/probe_speedl_stream.py` 파일 최상단(`"""..."""` 모듈 docstring 바로 뒤, `from collections import namedtuple` 앞)에 다음 두 줄을 추가:

```python
import argparse
import sys
```

그리고 `build_phase_plan` 정의 뒤에 아래 함수를 추가:

```python
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
```

- [ ] **Step 2: 수동 스모크 검증**

Run: `cd src/robot_control && PYTHONPATH="$(pwd)" python3 -c "from robot_control.tools.probe_speedl_stream import _parse_args; a = _parse_args(['--grasp-width-mm', '30', '--grasp-force-n', '20']); print(a)"`
Expected: `Namespace(robot_id='dsr01', axis='x', vel_mm_s=20.0, ...)` 형태로 출력되고 예외 없음.

Run: `cd src/robot_control && PYTHONPATH="$(pwd)" python3 -c "from robot_control.tools.probe_speedl_stream import _parse_args; _parse_args([])"`
Expected: `error: the following arguments are required: --grasp-width-mm, --grasp-force-n` 메시지와 함께 `SystemExit`

- [ ] **Step 3: 커밋**

```bash
git add src/robot_control/robot_control/tools/probe_speedl_stream.py
git commit -m "feat(robot_control): add CLI args and confirmation gate to speedl_stream probe"
```

---

### Task 3: 속도 벡터 계산 + 세그먼트 실행/정지 헬퍼

**Files:**
- Modify: `src/robot_control/robot_control/tools/probe_speedl_stream.py`

**Interfaces:**
- Consumes: `AXIS_INDEX`, `PhaseSegment`(Task 1)
- Produces: `_velocity_vector(axis, vel_mm_s, sign) -> list[float]`(길이 6); `_sleep_ticked(stop_event, seconds) -> bool`; `_publish_zero(pub, speedl_stream_cls, acc, period_s, repeats=5) -> None`; `_run_publish_segment(pub, speedl_stream_cls, axis, vel_mm_s, acc, period_s, segment, stop_event) -> bool`; `_run_phase_segments(pub, speedl_stream_cls, axis, vel_mm_s, acc, period_s, segments, stop_event) -> bool`; 상수 `STOP_TICK_S`, `FINAL_STOP_REPEATS`

스펙에 따라 이 함수들도 자동 테스트 대상이 아니다(실제 퍼블리셔/타이밍 의존) — Step 2에서 fake 퍼블리셔로 수동 스모크 검증한다.

- [ ] **Step 1: 헬퍼 함수 추가**

`src/robot_control/robot_control/tools/probe_speedl_stream.py` 파일 최상단 import 블록(`import sys` 다음)에 다음 두 줄을 추가:

```python
import threading
import time
```

그리고 `_confirm_or_exit` 뒤에 아래를 추가:

```python
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
```

- [ ] **Step 2: fake 퍼블리셔로 수동 스모크 검증**

Run:
```bash
cd src/robot_control && PYTHONPATH="$(pwd)" python3 -c "
import threading
from robot_control.tools.probe_speedl_stream import (
    build_phase_plan, _run_phase_segments)


class FakeMsg:
    def __init__(self):
        self.vel = None
        self.acc = None
        self.time = None


class FakePub:
    def __init__(self):
        self.calls = 0

    def publish(self, msg):
        self.calls += 1


pub = FakePub()
segments = build_phase_plan(
    phase_duration_s=0.1, osc_period_s=0.1, osc_duration_s=0.2,
    pause_durations_s=[0.1])
ok = _run_phase_segments(
    pub, FakeMsg, 'x', 20.0, (100.0, 100.0), 0.02, segments,
    threading.Event())
print('completed:', ok, 'publish calls:', pub.calls)
"
```
Expected: `completed: True publish calls: <0보다 큰 정수>` — 예외 없이 출력됨 (pause 구간에서는 publish가 호출되지 않으므로 calls는 segment들의 publish 구간 총 시간을 period_s로 나눈 근사치).

- [ ] **Step 3: 커밋**

```bash
git add src/robot_control/robot_control/tools/probe_speedl_stream.py
git commit -m "feat(robot_control): add velocity/publish/stop helpers to speedl_stream probe"
```

---

### Task 4: 이동 중 그리퍼 동작 확인 + main() 조립

**Files:**
- Modify: `src/robot_control/robot_control/tools/probe_speedl_stream.py`

**Interfaces:**
- Consumes: `build_phase_plan`(Task 1), `_parse_args`/`_confirm_or_exit`(Task 2), `_velocity_vector`/`_publish_zero`/`_run_publish_segment`/`_run_phase_segments`(Task 3), `robot_control.rg2_client.RG2Client`(기존 — `close(width_mm, force_n, goal_handle=None) -> bool`, `open(goal_handle=None) -> bool`, `last_status` 속성)
- Produces: `_run_gripper_during_motion(pub, speedl_stream_cls, axis, vel_mm_s, acc, period_s, rg2_client, grasp_width_mm, grasp_force_n, stop_event) -> None`; `main(argv=None) -> None`

- [ ] **Step 1: 그리퍼 동시 실행 헬퍼 + main() 추가**

`src/robot_control/robot_control/tools/probe_speedl_stream.py` 파일 최상단 import 블록(`import threading`/`import time` 다음)에 다음을 추가:

```python
import signal
from collections import namedtuple  # noqa: F811 (이미 있으면 중복 추가하지 않는다)

from robot_control.rg2_client import RG2Client
```

(`from collections import namedtuple`는 Task 1에서 이미 추가했으므로 실제로는 `import signal`과 `from robot_control.rg2_client import RG2Client` 두 줄만 새로 추가하면 된다.)

그리고 파일 하단(`_confirm_or_exit`/헬퍼 함수들 뒤)에 아래를 추가:

```python
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
    pub = node.create_publisher(SpeedlStream, f'/{args.robot_id}/speedl_stream', 10)
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
        _publish_zero(pub, SpeedlStream, acc, args.period_s)

        if completed and not stop_event.is_set():
            _run_gripper_during_motion(
                pub, SpeedlStream, args.axis, args.vel_mm_s, acc,
                args.period_s, rg2_client, args.grasp_width_mm,
                args.grasp_force_n, stop_event)

        print('--- phase6_final_stop ---')
        _publish_zero(pub, SpeedlStream, acc, args.period_s)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

이 시점에 파일 최상단 import 블록은 다음과 같은 상태여야 한다(Task 1~4에서 누적):

```python
import argparse
import signal
import sys
import threading
import time
from collections import namedtuple

from robot_control.rg2_client import RG2Client
```

- [ ] **Step 2: 문법/임포트 스모크 검증 (rclpy/dsr_msgs2 없이)**

Run: `cd src/robot_control && PYTHONPATH="$(pwd)" python3 -c "import robot_control.tools.probe_speedl_stream as m; print(m.main)"`
Expected: 예외 없이 `<function main at 0x...>` 출력 (rclpy/dsr_msgs2는 `main()` 내부에서만 import되므로 모듈 자체는 두 라이브러리 없이도 import 가능해야 함).

Run: `cd src/robot_control && PYTHONPATH="$(pwd)" python3 -m py_compile robot_control/tools/probe_speedl_stream.py`
Expected: 출력 없음(문법 오류 없음)

- [ ] **Step 3: 커밋**

```bash
git add src/robot_control/robot_control/tools/probe_speedl_stream.py
git commit -m "feat(robot_control): wire main() and gripper-during-motion check into speedl_stream probe"
```

---

### Task 5: entry_point 등록 및 최종 확인

**Files:**
- Modify: `src/robot_control/setup.py:26` (entry_points.console_scripts)

**Interfaces:**
- Consumes: `main`(Task 4)

- [ ] **Step 1: entry_point 추가**

`src/robot_control/setup.py`의 `entry_points` 블록을 다음과 같이 수정:

```python
    entry_points={
        'console_scripts': [
            'robot_control_node = robot_control.robot_control_node:main',
            'probe_speedl_stream = robot_control.tools.probe_speedl_stream:main',
        ],
    },
```

- [ ] **Step 2: --help 스모크 검증**

Run: `cd src/robot_control && PYTHONPATH="$(pwd)" python3 -m robot_control.tools.probe_speedl_stream --help`
Expected: argparse 도움말 출력(`--robot-id`, `--axis`, `--grasp-width-mm` 등 모든 옵션 나열), exit code 0.

- [ ] **Step 3: 전체 테스트 재확인**

Run: `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd)" python3 -m pytest test/test_probe_speedl_stream.py -v`
Expected: `5 passed` (Task 1 이후 변경으로 깨진 게 없는지 최종 확인)

- [ ] **Step 4: 커밋**

```bash
git add src/robot_control/setup.py
git commit -m "feat(robot_control): register probe_speedl_stream console script entry point"
```

---

## 실제 하드웨어 실행 방법 (참고 — 이 플랜의 작업 범위 밖)

워크스페이스 빌드/소싱 후:
```bash
ros2 run robot_control probe_speedl_stream --grasp-width-mm 30 --grasp-force-n 20
```
기본값(20mm/s, 3~5초 구간)으로 실행되며, Enable 스위치를 쥔 상태에서 `yes`를 입력해야 실제로 움직이기 시작한다.
