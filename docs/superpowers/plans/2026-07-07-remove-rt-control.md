# RT 세션 제거 및 비-RT speedl_stream 전환 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `robot_control`의 `servo_pick`/`handover_approach`에서 두산 RT 제어 세션(`ConnectRtControl`/`StartRtControl`/`StopRtControl`/`DisconnectRtControl` + `SpeedlRtStream`)을 완전히 제거하고, 비-RT `speedl_stream`(`SpeedlStream`)으로 전환한다. RT가 제공하던 "명령 끊기면 자동 정지"는 별도 스레드로 도는 데드맨스위치 워치독(`SpeedlWatchdog`)으로 대체한다.

**Architecture:** `_run_rt_tracking`(task_executor.py, servo_pick·handover_approach 공유)에서 RT 세션 열기/닫기 로직을 제거하고, `doosan_driver.py`의 `publish_speedl_rt()`를 `publish_speedl(command, accel_param_prefix, period_param_name)`으로 교체한다(기존에 handover_approach 호출 시에도 servo_pick의 acc/period 파라미터를 쓰던 교차배선 버그를 이 기회에 바로잡는다). 새 `SpeedlWatchdog` 클래스(rclpy 비의존, 순수 파이썬)가 메인 루프와 별개 스레드에서 돌며, 루프가 매 틱 `pet()`하지 않으면 `watchdog_timeout_s`(0.2초) 후 독립적으로 `vel=0`을 발행한다(2026-07-07 `probe_speedl_stream.py` 실측: 단일 정지 명령으로 충분히 멈추고 유지됨을 확인).

**Tech Stack:** rclpy, dsr_msgs2(SpeedlStream), Python threading, pytest.

## Global Constraints

- 워치독은 같은 프로세스 내 데드맨스위치 스레드다 — 프로세스 자체가 죽는 경우(kill -9, segfault)는 보호 범위 밖이다(사용자 확인됨, 범위 밖).
- 워치독 데드라인: `0.2`초 (기존 루프 주기 `0.01`초의 20배).
- `servo_pick`, `handover_approach` 둘 다 전환한다(`_run_rt_tracking` 공유).
- `handover_hold`(compliance 기반)는 RT 스트리밍과 무관 — 변경 없음.
- `servo_pick.hardware_ready`/`handover_approach.hardware_ready` 게이트는 유지한다 — 좌표계/TCP 오프셋 미검증이라는 별개 이유가 남아있다. 관련 주석에서 "RT" 언급만 제거한다.
- `publish_speedl`은 호출자(servo_pick 또는 handover_approach)에 맞는 자신의 acc/period 파라미터를 쓰도록 고친다(기존 교차배선 버그 수정, 사용자 승인됨).
- 새 파라미터 기본값: `speedl_acc_trans_mm_s2=200.0`, `speedl_acc_rot_deg_s2=60.0`(기존 6원소 `speedl_acc` 기본값 `[200,200,200,60,60,60]`에서 병진/회전 대표값을 그대로 가져옴 — 추측 아님), `watchdog_timeout_s=0.2`.
- RT 세션 실패를 다루던 테스트는 삭제한다(해당 실패 모드 자체가 더 이상 존재하지 않음). RT 세션 정리를 검증하던 테스트는 그 의도(취소/예외 시 정상 종료)만 남기고 RT 관련 추적을 제거한다.
- 테스트 실행 명령: `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd)" python3 -m pytest test/ -v` (환경의 pytest가 `anyio` 플러그인과 충돌하므로 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 필요 — 기존 확인됨).

---

### Task 1: `SpeedlWatchdog` 데드맨스위치 (신규, 독립)

**Files:**
- Create: `src/robot_control/robot_control/speedl_watchdog.py`
- Test: `src/robot_control/test/test_speedl_watchdog.py`

**Interfaces:**
- Produces: `class SpeedlWatchdog: __init__(self, timeout_s, on_timeout, poll_interval_s=0.05)`, `.start() -> None`, `.pet() -> None`, `.stop() -> None`.

- [ ] **Step 1: 실패하는 테스트 작성**

`src/robot_control/test/test_speedl_watchdog.py` 생성:

```python
import time

from robot_control.speedl_watchdog import SpeedlWatchdog


def test_pet_prevents_timeout():
    calls = []
    wd = SpeedlWatchdog(
        timeout_s=0.1, on_timeout=lambda: calls.append(True), poll_interval_s=0.02)
    wd.start()
    try:
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            wd.pet()
            time.sleep(0.02)
    finally:
        wd.stop()
    assert calls == []


def test_timeout_fires_once_when_pet_stops():
    calls = []
    wd = SpeedlWatchdog(
        timeout_s=0.05, on_timeout=lambda: calls.append(True), poll_interval_s=0.01)
    wd.start()
    time.sleep(0.2)
    wd.stop()
    assert calls == [True]


def test_stop_prevents_timeout_after_pet_stops():
    calls = []
    wd = SpeedlWatchdog(
        timeout_s=0.2, on_timeout=lambda: calls.append(True), poll_interval_s=0.02)
    wd.start()
    wd.stop()
    time.sleep(0.3)
    assert calls == []


def test_stop_without_start_does_not_raise():
    wd = SpeedlWatchdog(timeout_s=0.1, on_timeout=lambda: None)
    wd.stop()  # start() 없이 stop()만 호출해도 예외 없이 안전해야 한다
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd)" python3 -m pytest test/test_speedl_watchdog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robot_control.speedl_watchdog'`

- [ ] **Step 3: 최소 구현 작성**

`src/robot_control/robot_control/speedl_watchdog.py` 생성:

```python
import threading
import time


class SpeedlWatchdog:
    """메인 서보 루프와 별개 스레드에서 도는 데드맨스위치.

    루프가 pet()을 timeout_s 이내에 호출하지 않으면(예외로 루프가 죽거나 행이
    걸린 경우) 워치독 스레드가 독립적으로 on_timeout()을 호출한다. rclpy에
    의존하지 않는 순수 파이썬 클래스라 하드웨어 없이 유닛 테스트 가능하다.

    비-RT speedl은 명령이 끊겨도 스스로 멈추지 않지만(2026-07-07
    probe_speedl_stream.py 실측 확인), vel=0을 단 한 번만 발행해도 로봇이 멈춘
    채 유지된다(같은 실측) - 그래서 on_timeout은 한 번만 호출하고 스레드를
    종료한다(재시작하려면 start()를 다시 호출).

    같은 프로세스 내 스레드 기반이라 메인 루프가 행(hang)에 걸리거나 예외로
    죽어도 동작하지만, 프로세스 자체가 죽는 경우(kill -9, segfault)는 보호
    범위 밖이다.
    """

    def __init__(self, timeout_s, on_timeout, poll_interval_s=0.05):
        self._timeout_s = timeout_s
        self._on_timeout = on_timeout
        self._poll_interval_s = poll_interval_s
        self._last_pet = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._last_pet = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pet(self) -> None:
        self._last_pet = time.monotonic()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_s + self._poll_interval_s + 1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if time.monotonic() - self._last_pet > self._timeout_s:
                self._on_timeout()
                return
            time.sleep(self._poll_interval_s)


__all__ = ['SpeedlWatchdog']
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd)" python3 -m pytest test/test_speedl_watchdog.py -v`
Expected: `4 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/robot_control/robot_control/speedl_watchdog.py src/robot_control/test/test_speedl_watchdog.py
git commit -m "feat(robot_control): add SpeedlWatchdog dead-man's-switch for non-RT speedl"
```

---

### Task 2: `doosan_driver.py` — RT 제거, `publish_speedl` 추가

**Files:**
- Modify: `src/robot_control/robot_control/doosan_driver.py`

**Interfaces:**
- Produces: `DoosanDriver.publish_speedl(self, command, *, accel_param_prefix, period_param_name) -> None`
- Removes: `DoosanDriver.open_rt_session`, `DoosanDriver.close_rt_session`, `DoosanDriver.publish_speedl_rt`

- [ ] **Step 1: import 블록에서 RT 서비스/메시지 제거, SpeedlStream 추가**

`src/robot_control/robot_control/doosan_driver.py`에서:

```python
            from dsr_msgs2.srv import (
                ConnectRtControl,
                DisconnectRtControl,
                GetCurrentPosx,
                GetExternalTorque,
                GetRobotState,
                GetToolForce,
                MoveJoint,
                MoveLine,
                MoveStop,
                ReleaseComplianceCtrl,
                SetRobotControl,
                StartRtControl,
                StopRtControl,
                TaskComplianceCtrl,
            )
            from dsr_msgs2.msg import SpeedlRtStream
```

를 다음으로 교체:

```python
            from dsr_msgs2.srv import (
                GetCurrentPosx,
                GetExternalTorque,
                GetRobotState,
                GetToolForce,
                MoveJoint,
                MoveLine,
                MoveStop,
                ReleaseComplianceCtrl,
                SetRobotControl,
                TaskComplianceCtrl,
            )
            from dsr_msgs2.msg import SpeedlStream
```

- [ ] **Step 2: 저장된 클래스 참조에서 RT 관련 항목 제거**

다음을 찾아:

```python
        self._MoveJoint = MoveJoint
        self._MoveLine = MoveLine
        self._MoveStop = MoveStop
        self._GetRobotState = GetRobotState
        self._SetRobotControl = SetRobotControl
        self._GetExternalTorque = GetExternalTorque
        self._GetToolForce = GetToolForce
        self._GetCurrentPosx = GetCurrentPosx
        self._ConnectRtControl = ConnectRtControl
        self._StartRtControl = StartRtControl
        self._StopRtControl = StopRtControl
        self._DisconnectRtControl = DisconnectRtControl
        self._TaskComplianceCtrl = TaskComplianceCtrl
        self._ReleaseComplianceCtrl = ReleaseComplianceCtrl
        self._SpeedlRtStream = SpeedlRtStream
```

다음으로 교체:

```python
        self._MoveJoint = MoveJoint
        self._MoveLine = MoveLine
        self._MoveStop = MoveStop
        self._GetRobotState = GetRobotState
        self._SetRobotControl = SetRobotControl
        self._GetExternalTorque = GetExternalTorque
        self._GetToolForce = GetToolForce
        self._GetCurrentPosx = GetCurrentPosx
        self._TaskComplianceCtrl = TaskComplianceCtrl
        self._ReleaseComplianceCtrl = ReleaseComplianceCtrl
        self._SpeedlStream = SpeedlStream
```

- [ ] **Step 3: RT 서비스 클라이언트 생성 제거**

다음 블록을 찾아:

```python
        self._cli_connect_rt = node.create_client(
            ConnectRtControl, f'{prefix}/realtime/connect_rt_control',
            callback_group=group)
        self._cli_start_rt = node.create_client(
            StartRtControl, f'{prefix}/realtime/start_rt_control', callback_group=group)
        self._cli_stop_rt = node.create_client(
            StopRtControl, f'{prefix}/realtime/stop_rt_control', callback_group=group)
        self._cli_disconnect_rt = node.create_client(
            DisconnectRtControl, f'{prefix}/realtime/disconnect_rt_control',
            callback_group=group)
        self._cli_task_compliance = node.create_client(
```

다음으로 교체(RT 클라이언트 4개 삭제, task_compliance 클라이언트는 그대로 유지):

```python
        self._cli_task_compliance = node.create_client(
```

- [ ] **Step 4: 퍼블리셔를 speedl_stream(비-RT)으로 교체**

다음을 찾아:

```python
        self._pub_speedl_rt = node.create_publisher(
            SpeedlRtStream, f'{prefix}/speedl_rt_stream', 10)
```

다음으로 교체:

```python
        self._pub_speedl = node.create_publisher(
            SpeedlStream, f'{prefix}/speedl_stream', 10)
```

- [ ] **Step 5: `open_rt_session`/`close_rt_session` 메서드 제거**

다음 메서드 전체를 찾아 삭제:

```python
    def open_rt_session(self) -> bool:
        if self._cli_connect_rt.wait_for_service(timeout_sec=1.0):
            request = self._ConnectRtControl.Request()
            request.ip_address = self._node.get_parameter('servo_pick.rt_ip').value
            request.port = int(self._node.get_parameter('servo_pick.rt_port').value)
            response = self._wait_for_future(
                self._cli_connect_rt.call_async(request), 3.0)
            if not self._response_success(response):
                self._node.get_logger().warn(
                    'connect_rt_control 실패: 기존 드라이버 연결 여부를 확인하세요.')
        else:
            self._node.get_logger().warn('connect_rt_control 서비스 없음')

        if not self._cli_start_rt.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('start_rt_control 서비스 연결 실패')
        response = self._wait_for_future(
            self._cli_start_rt.call_async(self._StartRtControl.Request()), 3.0)
        if not self._response_success(response):
            raise RuntimeError('start_rt_control 실패')
        return True

    def close_rt_session(self) -> bool:
        stop_ok = False
        if self._cli_stop_rt.wait_for_service(timeout_sec=1.0):
            response = self._wait_for_future(
                self._cli_stop_rt.call_async(self._StopRtControl.Request()), 3.0)
            stop_ok = self._response_success(response)
        else:
            self._node.get_logger().error('stop_rt_control 서비스 연결 실패')

        disconnect_ok = False
        if self._cli_disconnect_rt.wait_for_service(timeout_sec=1.0):
            response = self._wait_for_future(
                self._cli_disconnect_rt.call_async(
                    self._DisconnectRtControl.Request()), 3.0)
            disconnect_ok = self._response_success(response)
        else:
            self._node.get_logger().error('disconnect_rt_control 서비스 연결 실패')
        return stop_ok and disconnect_ok

    def publish_speedl_rt(self, command):
        # 실제 단위가 확인될 때까지 hardware_ready 게이트로 발행을 막는다.
        message = self._SpeedlRtStream()
        message.vel = [
            command.vx, command.vy, command.vz, 0.0, 0.0, command.yaw_rate]
        message.acc = list(
            self._node.get_parameter('servo_pick.speedl_acc').value)
        message.time = self._node.get_parameter(
            'servo_pick.rt_control_period_s').value
        self._pub_speedl_rt.publish(message)
```

다음으로 교체:

```python
    def publish_speedl(self, command, *, accel_param_prefix, period_param_name):
        # 실제 단위가 확인될 때까지 hardware_ready 게이트로 발행을 막는다.
        # accel_param_prefix/period_param_name으로 호출자(servo_pick 또는
        # handover_approach)에 맞는 자신의 파라미터를 쓴다 - 예전에는 항상
        # servo_pick 것만 썼던 교차배선을 여기서 바로잡는다.
        message = self._SpeedlStream()
        message.vel = [
            command.vx, command.vy, command.vz, 0.0, 0.0, command.yaw_rate]
        message.acc = [
            self._node.get_parameter(f'{accel_param_prefix}.speedl_acc_trans_mm_s2').value,
            self._node.get_parameter(f'{accel_param_prefix}.speedl_acc_rot_deg_s2').value]
        message.time = self._node.get_parameter(period_param_name).value
        self._pub_speedl.publish(message)
```

- [ ] **Step 6: 문법 확인**

Run: `cd src/robot_control && python3 -m py_compile robot_control/doosan_driver.py`
Expected: 출력 없음(문법 오류 없음). `dsr_msgs2`가 설치되어 있지 않아도 이 파일은 함수 내부에서 지연 import하므로 `py_compile`은 통과해야 한다.

- [ ] **Step 7: 커밋**

```bash
git add src/robot_control/robot_control/doosan_driver.py
git commit -m "refactor(robot_control): replace RT session speedl_rt with non-RT speedl_stream in DoosanDriver"
```

---

### Task 3: 파라미터 변경 (`robot_control_node.py` + `robot_control_params.yaml`)

**Files:**
- Modify: `src/robot_control/robot_control/robot_control_node.py`
- Modify: `src/robot_control/config/robot_control_params.yaml`

**Interfaces:**
- Produces (신규/변경 파라미터): `servo_pick.control_period_s`, `servo_pick.speedl_acc_trans_mm_s2`, `servo_pick.speedl_acc_rot_deg_s2`, `servo_pick.watchdog_timeout_s`, `handover_approach.control_period_s`, `handover_approach.speedl_acc_trans_mm_s2`, `handover_approach.speedl_acc_rot_deg_s2`, `handover_approach.watchdog_timeout_s`
- Removes: `servo_pick.rt_ip`, `servo_pick.rt_port`, `servo_pick.rt_control_period_s`, `servo_pick.speedl_acc`, `handover_approach.rt_control_period_s`

- [ ] **Step 1: servo_pick 파라미터 블록 교체**

`src/robot_control/robot_control/robot_control_node.py`에서 다음을 찾아:

```python
        # servo_pick 실제 하드웨어 실행을 위한 별도 게이트. hardware_enabled=true여도
        # 이 값이 false면 servo_pick Goal 자체를 거부한다 (기본값 false).
        # 이유: 현재 ToolTrack.pose는 base_link 절대좌표로 정의되어 있는데
        # (handover_interfaces/msg/ToolTrack.msg), ServoLoop는 이를 TCP(그리퍼) 기준
        # xy 오차로 가정하고 P 제어를 수행한다 (servo_loop.py 상단 주석 참고). 이 좌표
        # 변환이 실제로 구현·검증되기 전까지는 실제 RT 속도 명령을 로봇에 보내면 안 된다.
        self.declare_parameter('servo_pick.hardware_ready', False)
        self.declare_parameter('servo_pick.rt_ip', '192.168.137.100')
        self.declare_parameter('servo_pick.rt_port', 12347)
        self.declare_parameter('servo_pick.rt_control_period_s', 0.01)
        _declare_double_array(
            self, 'servo_pick.speedl_acc', [200.0, 200.0, 200.0, 60.0, 60.0, 60.0])
```

다음으로 교체:

```python
        # servo_pick 실제 하드웨어 실행을 위한 별도 게이트. hardware_enabled=true여도
        # 이 값이 false면 servo_pick Goal 자체를 거부한다 (기본값 false).
        # 이유: 현재 ToolTrack.pose는 base_link 절대좌표로 정의되어 있는데
        # (handover_interfaces/msg/ToolTrack.msg), ServoLoop는 이를 TCP(그리퍼) 기준
        # xy 오차로 가정하고 P 제어를 수행한다 (servo_loop.py 상단 주석 참고). 이 좌표
        # 변환이 실제로 구현·검증되기 전까지는 실제 속도 명령을 로봇에 보내면 안 된다.
        self.declare_parameter('servo_pick.hardware_ready', False)
        self.declare_parameter('servo_pick.control_period_s', 0.01)
        self.declare_parameter('servo_pick.speedl_acc_trans_mm_s2', 200.0)
        self.declare_parameter('servo_pick.speedl_acc_rot_deg_s2', 60.0)
        # speedl(비-RT)은 명령이 끊겨도 스스로 멈추지 않는다(2026-07-07
        # probe_speedl_stream.py로 실측 확인) - SpeedlWatchdog가 이 시간 동안
        # pet()이 없으면 vel=0을 대신 발행한다. 단일 정지 명령으로 충분함도
        # 같은 실측으로 확인됨.
        self.declare_parameter('servo_pick.watchdog_timeout_s', 0.2)
```

- [ ] **Step 2: handover_approach 파라미터 블록 교체**

다음을 찾아:

```python
        self.declare_parameter('handover_approach.hardware_ready', False)
        # 사용자가 지정한 접근 정지 거리(5cm) - 실측 협의값.
        self.declare_parameter('handover_approach.stop_distance_m', 0.05)
        self.declare_parameter('handover_approach.v_max', 0.15)
        self.declare_parameter('handover_approach.kp_xy', 1.0)
        self.declare_parameter('handover_approach.timeout_s', 10.0)
        self.declare_parameter('handover_approach.t_lost_s', 0.5)
        self.declare_parameter('handover_approach.diverge_factor', 1.2)
        self.declare_parameter('handover_approach.diverge_window', 3)
        self.declare_parameter('handover_approach.rt_control_period_s', 0.01)
```

다음으로 교체:

```python
        self.declare_parameter('handover_approach.hardware_ready', False)
        # 사용자가 지정한 접근 정지 거리(5cm) - 실측 협의값.
        self.declare_parameter('handover_approach.stop_distance_m', 0.05)
        self.declare_parameter('handover_approach.v_max', 0.15)
        self.declare_parameter('handover_approach.kp_xy', 1.0)
        self.declare_parameter('handover_approach.timeout_s', 10.0)
        self.declare_parameter('handover_approach.t_lost_s', 0.5)
        self.declare_parameter('handover_approach.diverge_factor', 1.2)
        self.declare_parameter('handover_approach.diverge_window', 3)
        self.declare_parameter('handover_approach.control_period_s', 0.01)
        self.declare_parameter('handover_approach.speedl_acc_trans_mm_s2', 200.0)
        self.declare_parameter('handover_approach.speedl_acc_rot_deg_s2', 60.0)
        self.declare_parameter('handover_approach.watchdog_timeout_s', 0.2)
```

- [ ] **Step 3: handover_approach.hardware_ready 주석에서 "RT" 언급 제거**

다음을 찾아:

```python
        # hardware_ready는 servo_pick.hardware_ready와 같은 이유로 기본 false다:
        # hand_pose(vision_node._track_hand)가 아직 미구현(NotImplementedError)이라
        # frame_id/orientation 의미가 검증되지 않았다 - 확정 전까지 실제 RT 속도
        # 명령 발행을 금지한다.
```

다음으로 교체:

```python
        # hardware_ready는 servo_pick.hardware_ready와 같은 이유로 기본 false다:
        # hand_pose(vision_node._track_hand)가 아직 미구현(NotImplementedError)이라
        # frame_id/orientation 의미가 검증되지 않았다 - 확정 전까지 실제 속도
        # 명령 발행을 금지한다.
```

- [ ] **Step 4: `robot_control_params.yaml`의 "SpeedlRtStream 단위" 주석 갱신**

`src/robot_control/config/robot_control_params.yaml`에서 다음을 찾아:

```yaml
      # hardware_ready는 다음이 모두 확인되기 전까지 반드시 false로 유지한다:
      # ToolTrack.orientation 의미, TCP/그리퍼 offset, SpeedlRtStream 단위,
      # 실제 M0609 저속 검증. (기본값은 코드의 declare_parameter에 있으며 여기서는
      # 실수로 true를 채워 넣지 않기 위해 일부러 다시 적지 않는다.)
```

다음으로 교체:

```yaml
      # hardware_ready는 다음이 모두 확인되기 전까지 반드시 false로 유지한다:
      # ToolTrack.orientation 의미, TCP/그리퍼 offset, SpeedlStream 단위,
      # 실제 M0609 저속 검증. (기본값은 코드의 declare_parameter에 있으며 여기서는
      # 실수로 true를 채워 넣지 않기 위해 일부러 다시 적지 않는다.)
```

- [ ] **Step 5: 문법 확인**

Run: `cd src/robot_control && python3 -m py_compile robot_control/robot_control_node.py`
Expected: 출력 없음

- [ ] **Step 6: 커밋**

```bash
git add src/robot_control/robot_control/robot_control_node.py src/robot_control/config/robot_control_params.yaml
git commit -m "refactor(robot_control): rename RT-era params to non-RT names, add per-namespace speedl_acc/watchdog params"
```

---

### Task 4: `task_executor.py` — `_run_rt_tracking`에서 RT 제거 + 워치독 연결

**Files:**
- Modify: `src/robot_control/robot_control/task_executor.py`

**Interfaces:**
- Consumes: `SpeedlWatchdog`(Task 1), `DoosanDriver.publish_speedl`(Task 2), 새 파라미터명(Task 3)
- Removes: `TaskExecutor._open_rt_session`, `TaskExecutor._close_rt_session`, `TaskExecutor._cleanup_close_rt_session`
- Produces (시그니처 변경): `TaskExecutor._run_rt_tracking(self, goal_handle, *, name, message_type, topic, callback, servo, step, tick, validate_command, ready_parameter, period_parameter, accel_param_prefix)` — `accel_param_prefix` 신규 키워드 인자 추가.

- [ ] **Step 1: import에 `ServoCommand`, `SpeedlWatchdog` 추가**

`src/robot_control/robot_control/task_executor.py` 상단 import 블록에서 다음을 찾아:

```python
from robot_control.rg2_client import RG2Status
from robot_control.safety_monitor import FaultPrefix, SafetyState
```

다음으로 교체:

```python
from robot_control.rg2_client import RG2Status
from robot_control.safety_monitor import FaultPrefix, SafetyState
from robot_control.servo_loop import ServoCommand
from robot_control.speedl_watchdog import SpeedlWatchdog
```

- [ ] **Step 2: `_cleanup_close_rt_session` 메서드 삭제**

다음 메서드 전체를 찾아 삭제:

```python
    def _cleanup_close_rt_session(self) -> bool:
        """RT 세션 stop/disconnect (_close_rt_session 경계). _close_rt_session이
        실제로 확인한 성공 여부(bool)를 그대로 전달한다 - 예외 발생 시에도 절대
        상위로 전파하지 않는다."""
        try:
            return bool(self._close_rt_session())
        except Exception as exc:
            self.get_logger().error(f'RT 세션 종료 cleanup 중 예외: {exc}')
            return False

```

- [ ] **Step 3: `_open_rt_session`/`_close_rt_session` 메서드 삭제**

다음 메서드 전체를 찾아 삭제:

```python
    def _open_rt_session(self) -> bool:
        """RT 세션을 열고, 실제로 시작이 확인된 경우에만 True를 반환한다."""
        if not self.hardware_enabled:
            self.get_logger().info('[dry_run] RT 세션 오픈 생략')
            return True
        if self._doosan is None:
            raise RuntimeError('DoosanDriver가 초기화되지 않았습니다.')
        return self._doosan.open_rt_session()

    def _close_rt_session(self) -> bool:
        if not self.hardware_enabled:
            self.get_logger().info('[dry_run] RT 세션 종료 생략')
            return True
        if self._doosan is None:
            return True  # 정리할 DoosanDriver 자체가 없음 - idempotent하게 성공 취급
        return bool(self._doosan.close_rt_session())

```

- [ ] **Step 4: `_run_rt_tracking` 본문 교체**

다음 전체 메서드를 찾아:

```python
    def _run_rt_tracking(
            self, goal_handle, *, name, message_type, topic, callback,
            servo, step, tick, validate_command, ready_parameter, period_parameter):
        """물체·손 추적이 공통으로 사용하는 RT 실행/취소/정리 루프.

        step: 인자 없이 호출해 이번 틱의 ServoCommand(또는 아직 계산할 수 없으면
        None)를 반환하는 콜러블 - servo_pick(칼만 ServoLoop, TCP pose 필요)과
        handover_approach(HandApproachServo, 내부 상태만 사용)가 서로 다른
        step() 시그니처를 쓰므로 호출부에서 클로저로 그 차이를 흡수한다."""
        subscription = None
        rt_attempted = False
        outcome = 'ABORT'
        detail = f'{name} aborted'
        self._tcp_tracking_active = True
        try:
            rt_attempted = True
            rt_confirmed = self._open_rt_session()
            if self.hardware_enabled and not rt_confirmed:
                outcome, detail = 'ABORT', f'{name} aborted - RT 세션 시작 실패'
            else:
                subscription = self.create_subscription(
                    message_type, topic, callback, 10,
                    callback_group=self.sensor_callback_group)
                ready = bool(self.get_parameter(ready_parameter).value)
                period = float(self.get_parameter(period_parameter).value)

                while rclpy.ok():
                    if goal_handle.is_cancel_requested:
                        if not self._cleanup_stop_motion():
                            detail = f'{name} 취소 중 MoveStop 실패'
                            self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                            outcome = 'FAULT'
                        else:
                            outcome, detail = 'CANCELED', f'{name} canceled'
                        break
                    if self.safety_state != SafetyState.NORMAL:
                        outcome = 'ABORT'
                        detail = f'{name} aborted - safety_state={self.safety_state}'
                        break

                    state, reason = tick()
                    feedback = RobotTask.Feedback()
                    feedback.state = servo.get_state()
                    goal_handle.publish_feedback(feedback)
                    if state == 'ABORT':
                        outcome, detail = 'ABORT', reason
                        break
                    if state in ('CLOSE', 'STOP'):
                        outcome, detail = 'ARRIVED', ''
                        break

                    command = step()
                    if command is not None and (
                            self.hardware_enabled and ready and rt_confirmed
                            and self._doosan is not None):
                        if not validate_command(command):
                            outcome = 'ABORT'
                            detail = f'{name} aborted - invalid RT velocity command'
                            break
                        self._doosan.publish_speedl_rt(command)
                    time.sleep(period)
                else:
                    self._cleanup_stop_motion()
                    outcome, detail = 'ABORT', f'{name} aborted - rclpy 종료 중'
        except Exception as exc:
            self.get_logger().error(f'{name} 실행 중 예외: {exc}')
            stop_ok = self._cleanup_stop_motion()
            if goal_handle.is_cancel_requested and stop_ok:
                outcome, detail = 'CANCELED', f'{name} canceled after exception: {exc}'
            else:
                outcome, detail = 'ABORT', f'{name} exception: {exc}'
                if not stop_ok:
                    self._declare_fault(
                        f'{FaultPrefix.FAULT}{name} 예외 처리 중 MoveStop 실패: {exc}')
                    outcome = 'FAULT'
        finally:
            sub_ok = self._cleanup_destroy_subscription(subscription)
            rt_ok = self._cleanup_close_rt_session() if rt_attempted else True
            self._tcp_tracking_active = False
            if not (sub_ok and rt_ok):
                detail = (
                    f'{name} cleanup 실패 '
                    f'(subscription={sub_ok}, rt_session={rt_ok})')
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                outcome = 'FAULT'
        return outcome, detail
```

다음으로 교체:

```python
    def _run_rt_tracking(
            self, goal_handle, *, name, message_type, topic, callback,
            servo, step, tick, validate_command, ready_parameter,
            period_parameter, accel_param_prefix):
        """물체·손 추적이 공통으로 사용하는 실행/취소/정리 루프.

        RT 세션 없이 speedl_stream(비-RT)에 직접 연속 발행한다(2026-07-07
        probe_speedl_stream.py 실측: RT 세션 없이도 부드러운 서보잉이 가능함을
        확인). RT가 제공하던 "명령 끊기면 자동 정지"는 SpeedlWatchdog(데드맨
        스위치, 별도 스레드)로 대체한다 - 루프가 매 틱 pet()하고,
        watchdog_timeout_s 이내에 pet이 없으면 워치독이 독립적으로 vel=0을
        발행한다(단일 정지 명령으로 충분함도 같은 실측으로 확인됨).

        step: 인자 없이 호출해 이번 틱의 ServoCommand(또는 아직 계산할 수 없으면
        None)를 반환하는 콜러블 - servo_pick(칼만 ServoLoop, TCP pose 필요)과
        handover_approach(HandApproachServo, 내부 상태만 사용)가 서로 다른
        step() 시그니처를 쓰므로 호출부에서 클로저로 그 차이를 흡수한다."""
        subscription = None
        outcome = 'ABORT'
        detail = f'{name} aborted'
        self._tcp_tracking_active = True
        watchdog = SpeedlWatchdog(
            timeout_s=float(
                self.get_parameter(f'{accel_param_prefix}.watchdog_timeout_s').value),
            on_timeout=lambda: self._doosan.publish_speedl(
                ServoCommand(), accel_param_prefix=accel_param_prefix,
                period_param_name=period_parameter))
        try:
            subscription = self.create_subscription(
                message_type, topic, callback, 10,
                callback_group=self.sensor_callback_group)
            ready = bool(self.get_parameter(ready_parameter).value)
            period = float(self.get_parameter(period_parameter).value)
            publish_active = self.hardware_enabled and ready and self._doosan is not None
            if publish_active:
                watchdog.start()

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    if not self._cleanup_stop_motion():
                        detail = f'{name} 취소 중 MoveStop 실패'
                        self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                        outcome = 'FAULT'
                    else:
                        outcome, detail = 'CANCELED', f'{name} canceled'
                    break
                if self.safety_state != SafetyState.NORMAL:
                    outcome = 'ABORT'
                    detail = f'{name} aborted - safety_state={self.safety_state}'
                    break

                state, reason = tick()
                feedback = RobotTask.Feedback()
                feedback.state = servo.get_state()
                goal_handle.publish_feedback(feedback)
                if state == 'ABORT':
                    outcome, detail = 'ABORT', reason
                    break
                if state in ('CLOSE', 'STOP'):
                    outcome, detail = 'ARRIVED', ''
                    break

                command = step()
                if command is not None and publish_active:
                    if not validate_command(command):
                        outcome = 'ABORT'
                        detail = f'{name} aborted - invalid velocity command'
                        break
                    self._doosan.publish_speedl(
                        command, accel_param_prefix=accel_param_prefix,
                        period_param_name=period_parameter)
                    watchdog.pet()
                time.sleep(period)
            else:
                self._cleanup_stop_motion()
                outcome, detail = 'ABORT', f'{name} aborted - rclpy 종료 중'
        except Exception as exc:
            self.get_logger().error(f'{name} 실행 중 예외: {exc}')
            stop_ok = self._cleanup_stop_motion()
            if goal_handle.is_cancel_requested and stop_ok:
                outcome, detail = 'CANCELED', f'{name} canceled after exception: {exc}'
            else:
                outcome, detail = 'ABORT', f'{name} exception: {exc}'
                if not stop_ok:
                    self._declare_fault(
                        f'{FaultPrefix.FAULT}{name} 예외 처리 중 MoveStop 실패: {exc}')
                    outcome = 'FAULT'
        finally:
            watchdog.stop()
            sub_ok = self._cleanup_destroy_subscription(subscription)
            self._tcp_tracking_active = False
            if not sub_ok:
                detail = f'{name} cleanup 실패 (subscription={sub_ok})'
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                outcome = 'FAULT'
        return outcome, detail
```

- [ ] **Step 5: `_execute_servo_pick`의 `_run_rt_tracking` 호출부 갱신**

다음을 찾아:

```python
            ready_parameter='servo_pick.hardware_ready',
            period_parameter='servo_pick.rt_control_period_s')
```

다음으로 교체:

```python
            ready_parameter='servo_pick.hardware_ready',
            period_parameter='servo_pick.control_period_s',
            accel_param_prefix='servo_pick')
```

- [ ] **Step 6: `_execute_handover_approach`의 `_run_rt_tracking` 호출부 갱신**

다음을 찾아:

```python
            ready_parameter='handover_approach.hardware_ready',
            period_parameter='handover_approach.rt_control_period_s')
```

다음으로 교체:

```python
            ready_parameter='handover_approach.hardware_ready',
            period_parameter='handover_approach.control_period_s',
            accel_param_prefix='handover_approach')
```

- [ ] **Step 7: 문법 확인**

Run: `cd src/robot_control && python3 -m py_compile robot_control/task_executor.py`
Expected: 출력 없음

- [ ] **Step 8: 커밋**

```bash
git add src/robot_control/robot_control/task_executor.py
git commit -m "refactor(robot_control): remove RT session lifecycle from _run_rt_tracking, wire in SpeedlWatchdog"
```

---

### Task 5: `test_robot_control_node.py` 마이그레이션

**Files:**
- Modify: `src/robot_control/test/test_robot_control_node.py`

**Interfaces:**
- Consumes: Task 1~4에서 변경된 모든 인터페이스(`SpeedlWatchdog`, `DoosanDriver.publish_speedl`, `TaskExecutor._run_rt_tracking`의 새 시그니처, 새 파라미터명)

- [ ] **Step 1: `_FakeDoosanDriver` 갱신 — RT 관련 속성/메서드 제거, `publish_speedl`로 교체**

다음을 찾아:

```python
    def __init__(self):
        self.robot_state_sequence = []
        self.set_robot_control_calls = []
        self.ext_torque = [0.0] * 6
        self.tool_force = [0.0] * 6
        self.open_rt_session_should_fail = False
        self.publish_calls = []
        self.stop_calls = []
        self.stop_return_value = True
        self.stop_should_raise = False
        self.close_rt_session_should_raise = False
        self.close_rt_session_should_fail = False  # 예외 없이 응답 success=false만 시뮬레이션
```

다음으로 교체:

```python
    def __init__(self):
        self.robot_state_sequence = []
        self.set_robot_control_calls = []
        self.ext_torque = [0.0] * 6
        self.tool_force = [0.0] * 6
        self.publish_calls = []
        self.stop_calls = []
        self.stop_return_value = True
        self.stop_should_raise = False
```

다음을 찾아:

```python
    def open_rt_session(self):
        if self.open_rt_session_should_fail:
            raise RuntimeError('start_rt_control이 실패했습니다 (fake).')
        return True

    def close_rt_session(self):
        if self.close_rt_session_should_raise:
            raise RuntimeError('stop_rt_control 통신 오류 (fake).')
        if self.close_rt_session_should_fail:
            return False  # 예외 없이 StopRtControl/DisconnectRtControl 응답 success=false
        return True

    def stop(self, stop_mode=1):
```

다음으로 교체:

```python
    def stop(self, stop_mode=1):
```

다음을 찾아:

```python
    def publish_speedl_rt(self, cmd):
        self.publish_calls.append(cmd)
```

다음으로 교체:

```python
    def publish_speedl(self, cmd, *, accel_param_prefix, period_param_name):
        self.publish_calls.append(cmd)
```

- [ ] **Step 2: RT 세션 실패 전용 테스트 3개 삭제**

다음 함수 전체를 찾아 삭제(`test_servo_pick_rt_cleanup_failure_after_grasp_blocks_success`):

```python
def test_servo_pick_rt_cleanup_failure_after_grasp_blocks_success(node, monkeypatch):
    """필수 테스트 4: 파지 후 RT 세션 종료(cleanup) 실패 시 성공 처리를 금지한다."""
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._servo_pick_tick = lambda: ('CLOSE', None)
    node.servo_loop.get_state = lambda: 'closing'
    node.servo_loop.start = lambda *a, **k: None
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: (_ for _ in ()).throw(RuntimeError('rt close boom'))
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.succeeded is False
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


```

다음 함수 전체를 찾아 삭제(`test_servo_pick_cancel_aborts_as_fault_when_rt_close_raises`):

```python
def test_servo_pick_cancel_aborts_as_fault_when_rt_close_raises(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._open_rt_session = lambda: True
    node._close_rt_session = lambda: (_ for _ in ()).throw(RuntimeError('rt close boom'))
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is False
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


```

다음 함수 전체를 찾아 삭제(`test_servo_pick_cancel_aborts_as_fault_when_rt_close_response_unsuccessful`):

```python
def test_servo_pick_cancel_aborts_as_fault_when_rt_close_response_unsuccessful(node, monkeypatch):
    """필수 테스트: RT 종료(StopRtControl/DisconnectRtControl) 응답이 예외 없이
    success=false인 경우도 실패로 취급한다(예전처럼 응답을 확인하지 않고 조용히
    성공 취급하지 않는다)."""
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    fake.close_rt_session_should_fail = True
    node._doosan = fake
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is False
    assert gh.aborted is True
    assert _terminal_call_count(gh) == 1
    assert result.success is False
    assert node.safety_state == SafetyState.FAULT


```

다음 함수 전체를 찾아 삭제(`test_servo_pick_aborts_when_start_rt_control_fails`):

```python
def test_servo_pick_aborts_when_start_rt_control_fails(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('servo_pick.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    fake.open_rt_session_should_fail = True
    node._doosan = fake
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert fake.publish_calls == []  # RT가 시작되지 않았으므로 속도 명령을 보내지 않는다


```

다음 함수 전체를 찾아 삭제(`test_hardware_disabled_rt_session_is_noop`):

```python
def test_hardware_disabled_rt_session_is_noop(node):
    node._open_rt_session()
    node._close_rt_session()

    assert node._doosan is None
```

- [ ] **Step 3: RT 세션 정리 추적 테스트 2개를 취소-동작 검증으로 재정의**

다음을 찾아(`test_execute_handover_approach_cancel_mid_loop_calls_canceled_and_closes_rt_session`):

```python
def test_execute_handover_approach_cancel_mid_loop_calls_canceled_and_closes_rt_session(
        node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._handover_approach_tick = lambda: ('CONTINUE', None)
    node.hand_approach_servo.step = lambda: ServoCommand()
    node.hand_approach_servo.get_state = lambda: 'tracking'
    node.hand_approach_servo.start = lambda: None
    node._open_rt_session = lambda: None
    rt_closed = []
    node._close_rt_session = lambda: rt_closed.append(True) or True

    gh = FakeGoalHandle(_goal('handover_approach'))
    gh.is_cancel_requested = True

    result = node._execute_handover_approach(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
    assert rt_closed == [True]
```

다음으로 교체:

```python
def test_execute_handover_approach_cancel_mid_loop_calls_canceled(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._handover_approach_tick = lambda: ('CONTINUE', None)
    node.hand_approach_servo.step = lambda: ServoCommand()
    node.hand_approach_servo.get_state = lambda: 'tracking'
    node.hand_approach_servo.start = lambda: None

    gh = FakeGoalHandle(_goal('handover_approach'))
    gh.is_cancel_requested = True

    result = node._execute_handover_approach(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
```

다음을 찾아(`test_execute_servo_pick_cancel_mid_loop_calls_canceled_and_closes_rt_session`):

```python
def test_execute_servo_pick_cancel_mid_loop_calls_canceled_and_closes_rt_session(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    rt_closed = []
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: rt_closed.append(True) or True

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
    assert rt_closed == [True]  # finally에서 RT 세션 정리됨
```

다음으로 교체:

```python
def test_execute_servo_pick_cancel_mid_loop_calls_canceled(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0
    gh.is_cancel_requested = True

    result = node._execute_servo_pick(gh)

    assert gh.was_canceled is True
    assert gh.aborted is False
    assert result.success is False
```

- [ ] **Step 4: 이름에서 "rt_confirmed" 제거 (동작 변경 없음)**

다음을 찾아:

```python
def test_servo_pick_publishes_speedl_only_when_hardware_ready_and_rt_confirmed(node, monkeypatch):
```

다음으로 교체:

```python
def test_servo_pick_publishes_speedl_only_when_hardware_ready(node, monkeypatch):
```

- [ ] **Step 5: 섹션 헤더 주석 2곳 갱신**

다음을 찾아:

```python
# ---- SpeedlRtStream 발행 직전 마지막 안전 검사 (_validate_servo_command) ----
```

다음으로 교체:

```python
# ---- SpeedlStream 발행 직전 마지막 안전 검사 (_validate_servo_command) ----
```

다음을 찾아:

```python
# ---- servo_pick: cleanup(MoveStop/RT/subscription) 실패 시 취소 성공으로 가장하지 않음 ----
```

다음으로 교체:

```python
# ---- servo_pick: cleanup(MoveStop/subscription) 실패 시 취소 성공으로 가장하지 않음 ----
```

- [ ] **Step 6: 남은 단순 RT 세션 스텁 일괄 제거 (3개 패턴, `replace_all`)**

**패턴 A** (10곳 — `test_execute_handover_approach_success_stops_without_gripper_action`,
`test_execute_handover_approach_aborts_on_abort_reason`,
`test_execute_handover_approach_sets_and_clears_tcp_tracking_active`,
`test_execute_servo_pick_success_closes_gripper_and_returns_result`,
`test_servo_pick_rg2_canceled_during_close_ends_as_canceled_not_fault`,
`test_execute_servo_pick_abort_returns_reason`,
`test_execute_servo_pick_cancel_right_before_closing_gripper_does_not_close`,
그 외 동일 패턴 3곳): 아래 두 줄을 정확히 찾아(`replace_all=true`) 빈 문자열로 교체(즉 삭제):

```
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: True
```

**패턴 B** (2곳 — `test_execute_servo_pick_cancel_right_before_closing_gripper_does_not_close` 근처와 `_setup_servo_pick_dry` 헬퍼): 아래 두 줄을 정확히 찾아(`replace_all=true`) 빈 문자열로 교체:

```
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: None
```

**패턴 C** (1곳 — `test_servo_pick_cancel_aborts_as_fault_when_subscription_removal_fails`): 아래 두 줄을 찾아 빈 문자열로 교체:

```
    node._open_rt_session = lambda: True
    node._close_rt_session = lambda: True
```

각 교체 후 `grep -n "_open_rt_session\|_close_rt_session" src/robot_control/test/test_robot_control_node.py`로 잔여 참조가 없는지 확인한다(있다면 아직 처리 안 된 특수 케이스이니 Step 2~5에서 놓친 게 없는지 재확인).

- [ ] **Step 7: 워치독 통합 테스트 추가**

파일 끝(마지막 테스트 함수 뒤)에 추가:

```python


# ---- SpeedlWatchdog 통합 (명령이 끊기면 자동으로 vel=0 발행) ----

def test_servo_pick_watchdog_publishes_zero_when_no_command_computed(node, monkeypatch):
    """워치독 통합 테스트: hardware_ready 상태에서 step()이 계속 None을 반환해
    (tcp pose 미확보 등) pet()이 호출되지 않으면, watchdog_timeout_s 이내에
    워치독이 자동으로 vel=0 SpeedlStream을 발행한다(2026-07-07 실측: 단일
    정지 명령으로 충분함을 확인)."""
    node.hardware_enabled = True
    node.set_parameters([
        Parameter('servo_pick.hardware_ready', value=True),
        Parameter('servo_pick.watchdog_timeout_s', value=0.05),
        Parameter('servo_pick.control_period_s', value=0.01),
    ])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    node.servo_loop.start = lambda *a, **k: None
    node.servo_loop.get_state = lambda: 'tracking'
    node._get_current_tcp_posx = lambda: None  # step()이 항상 None을 반환하게 함

    started = time.monotonic()

    def _tick():
        # 워치독이 발동할 시간을 벌어준다(0.05s timeout보다 넉넉하게), 이후 종료.
        if time.monotonic() - started < 0.15:
            return ('CONTINUE', None)
        return ('CLOSE', None)

    node._servo_pick_tick = _tick
    node.rg2_client.close = lambda width, force, goal_handle=None: True
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert result.success is True
    assert any(
        cmd.vx == 0.0 and cmd.vy == 0.0 and cmd.vz == 0.0
        for cmd in fake.publish_calls)  # 워치독이 최소 1회 vel=0을 발행했다
```

- [ ] **Step 8: 전체 테스트 실행**

Run: `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd)" python3 -m pytest test/ -v 2>&1 | tail -60`
Expected: 모든 테스트 통과(0 failed). RT 관련 삭제로 이전보다 테스트 개수가 4개 적고(삭제 4개), 워치독 테스트 1개(Task 1의 4개 + 이 파일의 1개)만큼 늘어난다.

- [ ] **Step 9: 커밋**

```bash
git add src/robot_control/test/test_robot_control_node.py
git commit -m "test(robot_control): migrate test_robot_control_node.py off RT session, add watchdog integration test"
```

---

## 자기 점검(placeholder/모순/모호성)

- 모든 코드 블록이 완전한 실제 코드(TBD 없음).
- Task 순서(1 watchdog → 2 driver → 3 params → 4 task_executor → 5 tests)가 의존관계를 만족: task_executor는 watchdog/driver/params 이름을 모두 사용하므로 그 뒤에 옴, 테스트 마이그레이션은 전부 완료된 뒤 마지막에 검증.
- `publish_speedl`의 `accel_param_prefix`/`period_param_name` 시그니처가 doosan_driver.py(Task 2 생산)·task_executor.py(Task 4 소비)·test 파일의 `_FakeDoosanDriver`(Task 5) 세 곳에서 동일하게 일치함을 재확인함.
- Task 5의 "단순 스텁 제거" 패턴 3개가 Task 5의 "삭제/재정의" 대상(Step 2~4)과 텍스트가 겹치지 않음을 미리 확인함(순서상 Step 2~5를 먼저 처리한 뒤 Step 6에서 패턴 일괄 삭제를 하므로 안전).
