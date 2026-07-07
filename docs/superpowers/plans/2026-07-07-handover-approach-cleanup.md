# handover_approach RT/스트리밍 요소 제거 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `handover_approach`가 더 이상 `HandApproachServo`/`_run_rt_tracking`/`speedl_stream`을 쓰지 않게 하고, 실제 접근 로직(movel, 사용자가 직접 구현 예정) 자리에 TODO 스텁만 남긴다.

**Architecture:** `servo_loop.py`에서 `HandApproachServo`/`HandApproachState`를 완전히 삭제하고, `task_executor.py`의 관련 헬퍼 4개를 삭제하고 `_execute_handover_approach`를 게이트 체크만 남긴 TODO 스텁으로 축소한다. `robot_control_node.py`와 `robot_control_params.yaml`에서 대응 파라미터/초기화를 제거한다. `servo_pick`은 전혀 건드리지 않는다.

**Tech Stack:** Python, pytest, rclpy(간접 - 테스트 실행 시 소싱 필요).

## Global Constraints

- `servo_pick` 관련 코드/파라미터/테스트는 절대 건드리지 않는다.
- `handover_approach.hardware_ready`/`stop_distance_m`/`timeout_s`/`hand_pose_frame_id`는 유지한다(나중에 movel 구현 시 재사용).
- `robot_control_params.yaml`의 `v_max`/`kp_xy`/`t_lost_s` 오버라이드도 코드의 파라미터 삭제와 함께 제거한다(죽은 설정 방지).
- `_execute_handover_approach`는 게이트 통과 시 `_finish_tracking_result(goal_handle, 'ABORT', 'handover_approach not yet implemented')`을 반환하는 TODO 스텁으로 축소한다.
- 테스트 실행 명령: `source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash` 후 `cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd):$PYTHONPATH" python3 -m pytest test/ -v` (환경의 pytest가 `anyio` 플러그인과 충돌하므로 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 필요, `rclpy`/`handover_interfaces` import를 위해 ROS 소싱 필요).
- 모든 라인 번호는 커밋 `81648a24f4c9f5a13be8f3bf490bdfbcad3781d3` 기준 - 편집 전 반드시 해당 라인의 실제 내용이 아래 "before" 텍스트와 일치하는지 확인한다.

---

### Task 1: 프로덕션 코드 — HandApproachServo 및 관련 사용처 전부 제거

이 태스크는 4개 파일을 함께 바꾼다(서로 강하게 의존적 - `HandApproachServo` 클래스를 지우면서 그 사용처를 같은 커밋에서 지우지 않으면 import가 깨진다). 테스트 파일은 Task 2에서 다룬다(지금은 여전히 삭제된 심볼을 참조하므로 `pytest`가 아니라 `py_compile`로만 검증한다).

**Files:**
- Modify: `src/robot_control/robot_control/servo_loop.py`
- Modify: `src/robot_control/robot_control/task_executor.py`
- Modify: `src/robot_control/robot_control/robot_control_node.py`
- Modify: `src/robot_control/config/robot_control_params.yaml`

**Interfaces:**
- Removes: `HandApproachServo`, `HandApproachState`(servo_loop.py); `TaskExecutor._compute_hand_pose_tcp_offset`, `_validate_handover_approach_command`, `_on_hand_pose_during_approach`, `_handover_approach_tick`(task_executor.py); `RobotControlNode.hand_approach_servo` 속성.
- Produces: `TaskExecutor._execute_handover_approach(self, goal_handle)` — 시그니처는 동일하게 유지, 본문만 게이트+TODO로 축소.

- [ ] **Step 1: `servo_loop.py`에서 `HandApproachServo`/`HandApproachState` 삭제**

`src/robot_control/robot_control/servo_loop.py`는 현재 정확히 319줄이고, 219번째 줄(`        return None` — `ServoLoop.should_abort`의 마지막 줄)까지가 계속 남아야 할 내용이다. 220번째 줄부터 파일 끝(319번째 줄)까지가 `HandApproachState`/`HandApproachServo` 전체다. 다음 명령으로 파일을 219줄로 자른다:

```bash
head -n 219 src/robot_control/robot_control/servo_loop.py > /tmp/servo_loop_trimmed.py
mv /tmp/servo_loop_trimmed.py src/robot_control/robot_control/servo_loop.py
```

실행 전 반드시 확인: `sed -n '215,222p' src/robot_control/robot_control/servo_loop.py`가 다음과 정확히 일치해야 한다(일치하지 않으면 STOP하고 보고):
```python
                return 'diverging'
        # 참고: "공구가 방향 전환하여 시야 이탈 예상"(2.8절)은 판정 기준이 모호해
        # 과설계 우려가 있으므로 1차 구현 범위에서 제외했다. 실측 후 필요하면 추가한다.
        return None


class HandApproachState:
    TRACKING = 'tracking'
```

- [ ] **Step 2: `task_executor.py`에서 `PoseStamped` import 제거**

`src/robot_control/robot_control/task_executor.py`에서 다음 줄을 찾아:

```python
from geometry_msgs.msg import PoseStamped
```

이 줄을 완전히 삭제한다(다른 곳에서 `PoseStamped`를 쓰지 않으므로 - Step 3/4 완료 후 `grep -n "PoseStamped" src/robot_control/robot_control/task_executor.py`가 아무것도 반환하지 않아야 한다).

- [ ] **Step 3: 4개 헬퍼 메서드 삭제 + `_execute_handover_approach` 본문을 TODO 스텁으로 축소**

두 변경을 한 번에 적용한다(중간에 본문 없는 미완성 함수 상태를 만들지 않기
위해). 다음 전체 블록을 찾아(`_compute_hand_pose_tcp_offset`부터
`_execute_handover_approach`의 마지막 줄까지):

```python
    def _compute_hand_pose_tcp_offset(self, message):
        error = self._tcp_position_error(
            message,
            self.get_parameter('handover_approach.hand_pose_frame_id').value)
        if error is None:
            return None
        offset = PoseStamped()
        offset.header = message.header
        offset.pose.position.x, offset.pose.position.y, offset.pose.position.z = error
        offset.pose.orientation = message.pose.orientation
        return offset

    def _validate_handover_approach_command(self, cmd) -> bool:
        """속도 명령을 실제로 발행하기 직전 마지막 안전 검사(_validate_servo_command와
        동일한 목적). HandApproachServo는 3축 모두 같은 v_max로 클립하므로 여기서도
        축 구분 없이 하나의 한계로 검사한다."""
        values = (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)
        if not all(math.isfinite(v) for v in values):
            return False
        tol = 1e-6
        v_max = abs(self.get_parameter('handover_approach.v_max').value)
        return all(abs(v) <= v_max + tol for v in values)

    def _on_hand_pose_during_approach(self, msg):
        offset_msg = self._compute_hand_pose_tcp_offset(msg)
        if offset_msg is None:
            # frame_id 불일치, NaN/Inf, TCP 조회 실패/신선도 미달 - 이번 프레임은
            # 유실된 것처럼 취급한다(HandApproachServo.should_abort의 t_lost_s가 감지한다).
            return
        self.hand_approach_servo.on_hand_pose(offset_msg)

    def _handover_approach_tick(self):
        abort_reason = self.hand_approach_servo.should_abort()
        if abort_reason is not None:
            return ('ABORT', abort_reason)
        if self.hand_approach_servo.should_stop():
            return ('STOP', None)
        return ('CONTINUE', None)

    def _execute_handover_approach(self, goal_handle):
        if self.safety_state != SafetyState.NORMAL:
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                f'handover_approach rejected - safety_state={self.safety_state}')
        if (self.hardware_enabled
                and not self.get_parameter('handover_approach.hardware_ready').value):
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                'handover_approach rejected - handover_approach.hardware_ready=false')

        self.hand_approach_servo.start()
        outcome, detail = self._run_rt_tracking(
            goal_handle,
            name='handover_approach',
            message_type=PoseStamped,
            topic='/vision/hand_pose',
            callback=self._on_hand_pose_during_approach,
            servo=self.hand_approach_servo,
            step=self.hand_approach_servo.step,
            tick=self._handover_approach_tick,
            validate_command=self._validate_handover_approach_command,
            ready_parameter='handover_approach.hardware_ready',
            period_parameter='handover_approach.control_period_s',
            accel_param_prefix='handover_approach')
        if outcome == 'ARRIVED':
            detail = 'handover_approach arrived'
        return self._finish_tracking_result(goal_handle, outcome, detail)
```

다음으로 교체(4개 헬퍼는 완전히 삭제되고, `_execute_handover_approach`는 게이트
체크 + TODO 스텁만 남는다 - 한 번의 교체로 항상 유효한 문법 상태를 유지):

```python
    def _execute_handover_approach(self, goal_handle):
        if self.safety_state != SafetyState.NORMAL:
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                f'handover_approach rejected - safety_state={self.safety_state}')
        if (self.hardware_enabled
                and not self.get_parameter('handover_approach.hardware_ready').value):
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                'handover_approach rejected - handover_approach.hardware_ready=false')

        # TODO: movel 기반 단발성 접근 구현 예정 - RT/speedl_stream 미사용.
        # /vision/hand_pose를 한 번만 읽어 현재 TCP->손 방향으로
        # stop_distance_m만큼 못 미친 지점까지 movel로 이동한다(재계산 없음).
        return self._finish_tracking_result(
            goal_handle, 'ABORT', 'handover_approach not yet implemented')
```

- [ ] **Step 4: `robot_control_node.py`의 import 수정**

다음을 찾아:

```python
from robot_control.servo_loop import HandApproachServo, ServoLoop
```

다음으로 교체:

```python
from robot_control.servo_loop import ServoLoop
```

- [ ] **Step 5: `robot_control_node.py`의 handover_approach 파라미터 블록 축소**

다음을 찾아:

```python
        # handover_approach: handover_safe 도착 후 /vision/hand_pose(작업자 손 위치)를
        # 향해 servo_pick과 같은 PBVS 패턴으로 접근하다 stop_distance_m 이내가 되면
        # 멈춘다(그리퍼 동작 없음 - 이후 handover_hold가 당김을 기다린다). 사람에게
        # 접근하는 동작이라 servo_pick과 별도 네임스페이스로 두어 더 보수적으로
        # 튜닝할 수 있게 한다.
        # hardware_ready는 servo_pick.hardware_ready와 같은 이유로 기본 false다:
        # hand_pose(vision_node._track_hand)가 아직 미구현(NotImplementedError)이라
        # frame_id/orientation 의미가 검증되지 않았다 - 확정 전까지 실제 속도
        # 명령 발행을 금지한다.
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
        # hand_pose가 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id.
        # TF 변환이 구현되지 않았으므로 다른 frame_id는 거부한다
        # (_compute_hand_pose_tcp_offset).
        self.declare_parameter('handover_approach.hand_pose_frame_id', 'base_link')
```

다음으로 교체:

```python
        # handover_approach: handover_safe 도착 후 /vision/hand_pose(작업자 손 위치)를
        # 향해 접근하다 stop_distance_m 이내가 되면 멈춘다(그리퍼 동작 없음 - 이후
        # handover_hold가 당김을 기다린다). 실제 접근 로직(movel 기반 단발성 이동)은
        # 아직 구현 전이라 _execute_handover_approach는 게이트 체크 후 TODO를
        # 반환하는 스텁 상태다 - 아래 파라미터는 그 구현이 재사용할 것들이다.
        # hardware_ready는 servo_pick.hardware_ready와 같은 이유로 기본 false다:
        # hand_pose(vision_node._track_hand)가 아직 미구현(NotImplementedError)이라
        # frame_id/orientation 의미가 검증되지 않았다 - 확정 전까지 실제 속도
        # 명령 발행을 금지한다.
        self.declare_parameter('handover_approach.hardware_ready', False)
        # 사용자가 지정한 접근 정지 거리(5cm) - 실측 협의값.
        self.declare_parameter('handover_approach.stop_distance_m', 0.05)
        self.declare_parameter('handover_approach.timeout_s', 10.0)
        # hand_pose가 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id.
        # TF 변환이 구현되지 않았으므로 다른 frame_id는 거부한다.
        self.declare_parameter('handover_approach.hand_pose_frame_id', 'base_link')
```

- [ ] **Step 6: `robot_control_node.py`의 `hand_approach_servo` 초기화 삭제**

다음 블록 전체를 찾아:

```python
        self.hand_approach_servo = HandApproachServo(
            kp_xy=self.get_parameter('handover_approach.kp_xy').value,
            v_max=self.get_parameter('handover_approach.v_max').value,
            timeout_s=self.get_parameter('handover_approach.timeout_s').value,
            t_lost_s=self.get_parameter('handover_approach.t_lost_s').value,
            stop_distance_m=self.get_parameter('handover_approach.stop_distance_m').value,
            diverge_factor=self.get_parameter('handover_approach.diverge_factor').value,
            diverge_window=self.get_parameter('handover_approach.diverge_window').value,
        )

        # DoosanDriver 초기화 실패 시 즉시 FAULT를 선언해야 하므로, 발행자를 먼저 만든다.
```

다음으로 교체(초기화 블록만 삭제, 뒤따르는 주석/코드는 그대로):

```python
        # DoosanDriver 초기화 실패 시 즉시 FAULT를 선언해야 하므로, 발행자를 먼저 만든다.
```

- [ ] **Step 7: `robot_control_params.yaml`의 handover_approach 블록 정리**

`src/robot_control/config/robot_control_params.yaml`에서 다음을 찾아:

```yaml
    # handover_safe 도착 후 작업자 손(/vision/hand_pose)에 접근하는 서보잉 - 손이
    # 아직 정확히 확인되지 않은 상태이므로 servo_pick보다 v_max를 낮게 잡았다.
    handover_approach:
      hand_pose_frame_id: 'base_link'
      stop_distance_m: 0.05  # 사용자 확정값(5cm)
      v_max: 0.15
      kp_xy: 1.0
      timeout_s: 10.0
      t_lost_s: 0.5
      # hardware_ready는 vision_node._track_hand가 구현·검증되기 전까지 반드시
      # false로 유지한다(hand_pose의 frame_id/orientation 의미가 아직 미확정).
      # 기본값은 코드의 declare_parameter에 있으며 실수로 true를 적지 않기 위해
      # 여기서는 다시 명시하지 않는다.
```

다음으로 교체:

```yaml
    # handover_safe 도착 후 작업자 손(/vision/hand_pose)에 접근 - 실제 접근 로직
    # (movel 기반 단발성 이동)은 아직 구현 전이라 TODO 스텁 상태다.
    handover_approach:
      hand_pose_frame_id: 'base_link'
      stop_distance_m: 0.05  # 사용자 확정값(5cm)
      timeout_s: 10.0
      # hardware_ready는 vision_node._track_hand가 구현·검증되기 전까지 반드시
      # false로 유지한다(hand_pose의 frame_id/orientation 의미가 아직 미확정).
      # 기본값은 코드의 declare_parameter에 있으며 실수로 true를 적지 않기 위해
      # 여기서는 다시 명시하지 않는다.
```

- [ ] **Step 8: 문법 확인**

Run: `cd src/robot_control && python3 -m py_compile robot_control/servo_loop.py robot_control/task_executor.py robot_control/robot_control_node.py`
Expected: 출력 없음(문법 오류 없음)

Run: `python3 -c "import yaml; yaml.safe_load(open('src/robot_control/config/robot_control_params.yaml'))" `
Expected: 예외 없이 종료(yaml 문법 유효)

- [ ] **Step 9: 커밋**

```bash
git add src/robot_control/robot_control/servo_loop.py src/robot_control/robot_control/task_executor.py src/robot_control/robot_control/robot_control_node.py src/robot_control/config/robot_control_params.yaml
git commit -m "refactor(robot_control): remove HandApproachServo and streaming plumbing from handover_approach

handover_approach no longer needs continuous PBVS velocity tracking -
approaching a hand only needs a single movel to stop short of it, which
will be implemented separately. Reduce _execute_handover_approach to its
existing gate checks plus a TODO stub."
```

---

### Task 2: 테스트 마이그레이션

**Files:**
- Modify: `src/robot_control/test/test_servo_loop.py`
- Modify: `src/robot_control/test/test_robot_control_node.py`

**Interfaces:**
- Consumes: Task 1의 모든 변경사항(특히 `_execute_handover_approach`의 새 TODO 스텁 반환값).

- [ ] **Step 1: `test_servo_loop.py` — hand_approach 테스트 12개 + 헬퍼 2개 + import 정리**

`src/robot_control/test/test_servo_loop.py`는 현재 정확히 276줄이고, 159번째 줄(`    assert loop.should_abort() is None` — `test_should_abort_none_when_healthy`의 마지막 줄)까지가 계속 남아야 할 내용이다. 160번째 줄부터 파일 끝(276번째 줄)까지가 `HandApproachServo` 관련 테스트 12개 전체다. 다음 명령으로 파일을 159줄로 자른다:

```bash
head -n 159 src/robot_control/test/test_servo_loop.py > /tmp/test_servo_loop_trimmed.py
mv /tmp/test_servo_loop_trimmed.py src/robot_control/test/test_servo_loop.py
```

실행 전 반드시 확인: `sed -n '155,163p' src/robot_control/test/test_servo_loop.py`가 다음과 정확히 일치해야 한다(일치하지 않으면 STOP하고 보고):
```python
def test_should_abort_none_when_healthy():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05))
    assert loop.should_abort() is None


# ==== HandApproachServo (handover_approach) ====

def test_hand_approach_initial_state_is_tracking():
```

이제 truncate된 파일(159줄)에서 이어서, import 블록을 수정한다. 다음을 찾아:

```python
import time

import pytest
from geometry_msgs.msg import PoseStamped
from robot_control.servo_loop import (
    HandApproachServo,
    HandApproachState,
    ServoCommand,
    ServoLoop,
    ServoState,
)
```

다음으로 교체:

```python
import time

import pytest
from robot_control.servo_loop import ServoCommand, ServoLoop, ServoState
```

그다음 `_make_hand_servo`/`_hand_pose` 헬퍼를 제거한다. 다음을 찾아(주의: 이 블록은 `_make_loop` 함수 뒤, `test_initial_state_is_tracking` 앞에 있다):

```python
def _make_loop(**overrides):
    kwargs = dict(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                  eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                  dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                  innov_low=0.010, innov_high=0.040, w_alpha=1.0,
                  z_close=0.02, diverge_n=5, cov_threshold=0.5)
    kwargs.update(overrides)
    return ServoLoop(**kwargs)


def _make_hand_servo(**overrides):
    kwargs = dict(kp_xy=1.0, v_max=0.15, timeout_s=5.0, t_lost_s=0.3, stop_distance_m=0.05)
    kwargs.update(overrides)
    return HandApproachServo(**kwargs)


def _hand_pose(x=0.0, y=0.0, z=0.0):
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    return msg


def test_initial_state_is_tracking():
```

다음으로 교체:

```python
def _make_loop(**overrides):
    kwargs = dict(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                  eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                  dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                  innov_low=0.010, innov_high=0.040, w_alpha=1.0,
                  z_close=0.02, diverge_n=5, cov_threshold=0.5)
    kwargs.update(overrides)
    return ServoLoop(**kwargs)


def test_initial_state_is_tracking():
```

- [ ] **Step 2: `test_servo_loop.py` 단독 실행으로 확인**

Run: `source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash && cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd):$PYTHONPATH" python3 -m pytest test/test_servo_loop.py -v`
Expected: 전부 PASS(HandApproachServo 관련 테스트가 없어졌으므로 이전보다 12개 적은 수)

- [ ] **Step 3: `test_robot_control_node.py` — 손 위치 오차/tick/validate_command 섹션(6+3+3=12개 테스트 + 헬퍼 1개) 통째로 삭제**

`src/robot_control/test/test_robot_control_node.py`의 715번째 줄부터 829번째 줄까지가
`_hand_pose_msg` 헬퍼 + `_compute_hand_pose_tcp_offset`/`_on_hand_pose_during_approach`
테스트 6개 + `_validate_handover_approach_command` 테스트 3개 +
`_handover_approach_tick` 테스트 3개 전체다(712번째 줄에서 끝나는
`test_on_tool_track_during_servo_forwards_raw_message_to_servo_loop`은 servo_pick
테스트라 유지, 830번째 줄의 `# ---- handover_approach goal 거부 ----` 섹션도 유지).

실행 전 반드시 확인: `sed -n '710,716p' test/test_robot_control_node.py`가 다음과
정확히 일치해야 하고:
```python
    node._on_tool_track_during_servo(msg)

    assert received == [msg]


# ---- 작업자 손 위치 -> TCP 오차 계산 (_compute_hand_pose_tcp_offset) ----

def _hand_pose_msg(frame_id='base_link', x=1.5, y=0.2, z=0.4):
```
`sed -n '826,831p' test/test_robot_control_node.py`가 다음과 정확히 일치해야 한다:
```python
    assert status == 'ABORT'
    assert reason == 'diverged'


# ---- handover_approach goal 거부 (hardware_ready 게이트) ----

def test_goal_callback_rejects_handover_approach_when_hardware_ready_false(node):
```

두 확인이 모두 일치하면 다음 명령으로 715~829번째 줄을 삭제한다:

```bash
sed -i '715,829d' src/robot_control/test/test_robot_control_node.py
```

- [ ] **Step 4: `test_execute_handover_approach_*` 스트리밍 전제 테스트 3개(성공/abort/취소) 삭제 + TODO 스텁 테스트로 교체**

Step 3 실행 후 파일이 114줄 줄어들었으므로, 이 Step은 텍스트 매칭(Edit)으로 진행한다(라인 번호가 아니라 정확한 코드 블록으로 찾는다). 다음 전체 블록을 찾아:

```python
def test_execute_handover_approach_success_stops_without_gripper_action(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    ticks = iter(['CONTINUE', 'CONTINUE', 'STOP'])
    node._handover_approach_tick = lambda: (next(ticks), None)
    node.hand_approach_servo.step = lambda: ServoCommand()
    node.hand_approach_servo.get_state = lambda: 'tracking'
    node.hand_approach_servo.start = lambda: None

    rg2_calls = []
    node.rg2_client.open = lambda goal_handle=None: rg2_calls.append('open') or True
    node.rg2_client.close = lambda width, force, goal_handle=None: rg2_calls.append('close') or True

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.succeeded is True
    assert result.success is True
    assert rg2_calls == []  # 그리퍼 동작은 전혀 하지 않는다
    assert len(gh.feedback_msgs) == 3


def test_execute_handover_approach_aborts_on_abort_reason(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._handover_approach_tick = lambda: ('ABORT', 'lost')
    node.hand_approach_servo.get_state = lambda: 'tracking'
    node.hand_approach_servo.start = lambda: None

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'lost'


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


def test_execute_handover_approach_rejected_when_safety_state_not_normal(node):
```

다음으로 교체(3개 테스트 삭제, 유지할 `rejected_when_safety_state_not_normal`의 def 줄만 남김):

```python
def test_execute_handover_approach_rejected_when_safety_state_not_normal(node):
```

- [ ] **Step 5: `test_execute_handover_approach_sets_and_clears_tcp_tracking_active` 삭제 + TODO 스텁 검증 테스트 추가**

다음 전체 블록을 찾아:

```python
def test_execute_handover_approach_sets_and_clears_tcp_tracking_active(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    observed = {'active': None}

    def fake_tick():
        observed['active'] = node._tcp_tracking_active
        return ('STOP', None)

    node._handover_approach_tick = fake_tick
    node.hand_approach_servo.get_state = lambda: 'arrived'
    node.hand_approach_servo.start = lambda: None

    gh = FakeGoalHandle(_goal('handover_approach'))

    assert node._tcp_tracking_active is False
    node._execute_handover_approach(gh)

    assert observed['active'] is True
    assert node._tcp_tracking_active is False
```

다음으로 교체:

```python
def test_execute_handover_approach_returns_not_implemented_when_gates_pass(node):
    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'handover_approach not yet implemented'
```

- [ ] **Step 6: `test_handover_approach_publishes_speedl_with_own_param_prefix` 삭제**

다음 전체 블록을 찾아:

```python
def test_handover_approach_publishes_speedl_with_own_param_prefix(node, monkeypatch):
    """servo_pick과의 교차배선 버그 수정 검증: handover_approach 실행 시
    publish_speedl에 accel_param_prefix='handover_approach'/
    period_param_name='handover_approach.control_period_s'가 전달되는지 확인한다
    (예전에는 servo_pick의 파라미터를 항상 썼던 교차배선이 있었다)."""
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node.hardware_enabled = True
    node.set_parameters([Parameter('handover_approach.hardware_ready', value=True)])
    fake = _FakeDoosanDriver()
    node._doosan = fake
    ticks = iter(['CONTINUE', 'STOP'])
    node._handover_approach_tick = lambda: (next(ticks), None)
    node.hand_approach_servo.step = lambda: ServoCommand(vx=0.1)
    node.hand_approach_servo.start = lambda: None
    node.hand_approach_servo.get_state = lambda: 'tracking'

    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert result.success is True
    assert len(fake.publish_calls) == 1
    assert fake.publish_kwargs[0] == {
        'accel_param_prefix': 'handover_approach',
        'period_param_name': 'handover_approach.control_period_s',
    }


# ---- SpeedlWatchdog 통합 (명령이 끊기면 자동으로 vel=0 발행) ----
```

다음으로 교체(테스트만 삭제, 다음 섹션 헤더는 유지):

```python
# ---- SpeedlWatchdog 통합 (명령이 끊기면 자동으로 vel=0 발행) ----
```

- [ ] **Step 7: 잔여 참조 확인**

Run: `grep -n "hand_approach_servo\|HandApproachServo\|HandApproachState\|_handover_approach_tick\|_validate_handover_approach_command\|_compute_hand_pose_tcp_offset\|_on_hand_pose_during_approach\|_hand_pose_msg" src/robot_control/test/test_robot_control_node.py`
Expected: 출력 없음(잔여 참조 0건)

- [ ] **Step 8: 전체 테스트 스위트 실행**

Run: `source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash && cd src/robot_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$(pwd):$PYTHONPATH" python3 -m pytest test/ -v`
Expected: 전부 PASS, pristine 출력(경고 없음)

- [ ] **Step 9: 커밋**

```bash
git add src/robot_control/test/test_servo_loop.py src/robot_control/test/test_robot_control_node.py
git commit -m "test(robot_control): migrate tests off HandApproachServo, add TODO-stub coverage"
```

---

## 자기 점검(placeholder/모순/모호성)

- 모든 코드 블록이 완전한 실제 텍스트(TBD 없음), 각 sed/head 명령 전에 실행자가
  직접 확인할 "before" 텍스트를 명시해 라인 드리프트 위험을 낮춤.
- Task 1(프로덕션)이 Task 2(테스트)보다 먼저 와야 하는 이유(안 그러면 Task 1
  중간 상태에서 import가 깨짐)를 태스크 설명에 명시함.
- Task 2 Step 3(sed 삭제)과 Step 4~6(Edit 텍스트 매칭)의 순서 의존성을 명시함 -
  Step 3이 라인을 114줄 줄이므로 그 이후는 라인 번호가 아니라 텍스트로 찾아야 함.
- `test_servo_pick_*`/`test_execute_servo_pick_*` 등 servo_pick 관련 테스트는
  이 플랜의 어떤 Step에서도 건드리지 않음 - 범위 경계 확인됨.
- `robot_control_params.yaml`의 `v_max`/`kp_xy`/`t_lost_s` 제거가 Task 1의
  파라미터 삭제와 정확히 짝을 이룸(스펙 self-review에서 이미 확인됨).
