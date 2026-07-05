# 팀원2 (vision_tracking + calculate) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/전체 계획.md` 8절의 팀원2 역할(vision_node 추적 + calculate 모듈)을 기존 워크스페이스의
`NotImplementedError` 스텁에 채워 넣어, 하드웨어 없이도 시뮬레이션으로 검증 가능한 상태로 만든다.

**Architecture:** 알고리즘 핵심(칼만 필터, 추적기)은 rclpy에 의존하지 않는 순수 파이썬 모듈로 분리해
유닛테스트하고, ROS 노드 파일(`vision_node.py`, `servo_loop.py`, `robot_control_node.py`)은 그 모듈을
불러 쓰는 얇은 wiring 레이어로 유지한다.

**Tech Stack:** rclpy, message_filters, cv_bridge, numpy, mediapipe(손 추적), pytest.

## Global Constraints

- 설계 근거: `docs/전체 계획.md` 2절(제어 루프), 2.5절(필터), 2.6절(파지판정), 2.8절(abort),
  8절(역할 분담). 스펙: `docs/superpowers/specs/2026-07-05-teammate2-vision-calculate-design.md`
- YOLO 검출 자체는 팀원3 담당 — 이 계획은 검출 결과(bbox)를 입력으로만 소비한다
- yaw는 0 고정 (1차 범위 밖)
- Doosan RT 세션·TCP pose 조회·RG2 Modbus 통신은 팀원1 담당 — 이 계획에서는 `NotImplementedError`
  스텁으로만 남기고 호출 인터페이스만 맞춘다
- 기존 코드 스타일(들여쓰기 4칸, 클래스/함수 구조) 유지

---

### Task 1: `handover_interfaces`에 검출 인터페이스 msg 추가

**Files:**
- Create: `src/handover_interfaces/msg/Detection2D.msg`
- Create: `src/handover_interfaces/msg/DetectionArray.msg`
- Modify: `src/handover_interfaces/CMakeLists.txt`

**Interfaces:**
- Produces: `handover_interfaces.msg.Detection2D` (fields: `class_name:string, score:float32,
  x1/y1/x2/y2:int32`), `handover_interfaces.msg.DetectionArray` (fields: `header:std_msgs/Header,
  detections:Detection2D[]`) — Task 5, 7에서 사용

- [ ] **Step 1: msg 파일 작성**

`src/handover_interfaces/msg/Detection2D.msg`:
```
string class_name
float32 score
int32 x1
int32 y1
int32 x2
int32 y2
```

`src/handover_interfaces/msg/DetectionArray.msg`:
```
std_msgs/Header header
Detection2D[] detections
```

- [ ] **Step 2: CMakeLists.txt에 등록**

`src/handover_interfaces/CMakeLists.txt`의 `rosidl_generate_interfaces` 블록을 아래로 교체:
```cmake
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ToolTrack.msg"
  "msg/GripperState.msg"
  "msg/Detection2D.msg"
  "msg/DetectionArray.msg"
  "srv/SetVisionMode.srv"
  "action/RobotTask.action"
  DEPENDENCIES std_msgs geometry_msgs
)
```

- [ ] **Step 3: 빌드 및 임포트 확인**

Run:
```bash
cd /home/hwangjeongui/rokey_proj_02
colcon build --packages-select handover_interfaces
source install/setup.bash
python3 -c "from handover_interfaces.msg import Detection2D, DetectionArray; print('ok')"
```
Expected: `ok` 출력, 에러 없음

- [ ] **Step 4: 커밋**

```bash
git add src/handover_interfaces/msg/Detection2D.msg src/handover_interfaces/msg/DetectionArray.msg src/handover_interfaces/CMakeLists.txt
git commit -m "feat(handover_interfaces): add Detection2D/DetectionArray msgs for tool detection input"
```

---

### Task 2: 칼만 필터 (`kalman.py`)

**Files:**
- Create: `src/robot_control/robot_control/kalman.py`
- Test: `src/robot_control/test/test_kalman.py`

**Interfaces:**
- Produces: `KalmanXYZV(q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0)` with
  `.initialize(x,y,z)`, `.predict(dt)`, `.predict_position(lead_time) -> np.ndarray[3]`,
  `.update_xyz(meas_xyz) -> float(innov_xy)`, `.update_xy_only(meas_xy) -> float(innov_xy)`,
  `.reset_velocity_covariance()`, `.position -> np.ndarray[3]`, `.velocity -> np.ndarray[2]`,
  `.velocity_covariance_trace -> float`, `._initialized: bool` — Task 3에서 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`src/robot_control/test/test_kalman.py`:
```python
import pytest
from robot_control.kalman import KalmanXYZV


def test_initialize_sets_position_and_zero_velocity():
    kf = KalmanXYZV()
    kf.initialize(1.0, 2.0, 0.05)
    assert list(kf.position) == pytest.approx([1.0, 2.0, 0.05])
    assert list(kf.velocity) == pytest.approx([0.0, 0.0])
    assert kf._initialized is True


def test_predict_advances_position_by_velocity():
    kf = KalmanXYZV()
    kf.initialize(0.0, 0.0, 0.05)
    kf.x[3] = 0.1
    kf.predict(dt=1.0)
    assert kf.position[0] == pytest.approx(0.1, abs=1e-6)


def test_update_xyz_returns_innovation_and_pulls_toward_measurement():
    kf = KalmanXYZV(q_pos=1e-3, q_vel=1e-2, r_xy=1e-4, r_z=1e-4)
    kf.initialize(0.0, 0.0, 0.05)
    kf.predict(dt=0.02)
    innov = kf.update_xyz([0.002, 0.0, 0.05])
    assert innov == pytest.approx(0.002, abs=1e-6)
    assert 0.0 < kf.position[0] < 0.002


def test_update_xy_only_leaves_z_unchanged():
    kf = KalmanXYZV()
    kf.initialize(0.0, 0.0, 0.05)
    kf.predict(dt=0.02)
    kf.update_xy_only([0.001, 0.0])
    assert kf.position[2] == pytest.approx(0.05, abs=1e-9)


def test_reset_velocity_covariance_sets_trace_to_2x_p0():
    kf = KalmanXYZV(p0_vel_reset=2.0)
    kf.initialize(0.0, 0.0, 0.05)
    kf.predict(dt=0.02)
    kf.update_xyz([0.0, 0.0, 0.05])
    kf.reset_velocity_covariance()
    assert kf.velocity_covariance_trace == pytest.approx(4.0)


def test_predict_position_does_not_mutate_state():
    kf = KalmanXYZV()
    kf.initialize(0.0, 0.0, 0.05)
    kf.x[3] = 0.1
    before = kf.position.copy()
    p = kf.predict_position(0.5)
    assert list(kf.position) == pytest.approx(list(before))
    assert p[0] == pytest.approx(0.05)
    assert p[2] == pytest.approx(0.05)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m pytest test/test_kalman.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robot_control.kalman'`

- [ ] **Step 3: 구현**

`src/robot_control/robot_control/kalman.py`:
```python
import numpy as np


class KalmanXYZV:
    """base_link 좌표 공구 위치용 등속 모델 칼만 필터. 상태=[x,y,z,vx,vy] (전체 계획.md 2.5절)."""

    def __init__(self, q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0):
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_xy = r_xy
        self.r_z = r_z
        self.p0_vel_reset = p0_vel_reset
        self.x = np.zeros(5)
        self.P = np.eye(5)
        self._initialized = False

    def initialize(self, x, y, z):
        self.x = np.array([x, y, z, 0.0, 0.0])
        self.P = np.eye(5)
        self._initialized = True

    def _F(self, dt):
        F = np.eye(5)
        F[0, 3] = dt
        F[1, 4] = dt
        return F

    def _Q(self, dt):
        q = np.array([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel])
        return np.diag(q * max(dt, 1e-6))

    def predict(self, dt):
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def predict_position(self, lead_time):
        """상태를 바꾸지 않고 lead_time 이후 위치만 외삽해서 반환한다."""
        px, py, pz, vx, vy = self.x
        return np.array([px + vx * lead_time, py + vy * lead_time, pz])

    def update_xyz(self, meas_xyz):
        H = np.zeros((3, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = np.diag([self.r_xy, self.r_xy, self.r_z])
        return self._update(H, R, np.asarray(meas_xyz, dtype=float))

    def update_xy_only(self, meas_xy):
        H = np.zeros((2, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.diag([self.r_xy, self.r_xy])
        return self._update(H, R, np.asarray(meas_xy, dtype=float))

    def _update(self, H, R, z):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
        return float(np.linalg.norm(y[:2]))

    def reset_velocity_covariance(self):
        for i in (3, 4):
            self.P[i, :] = 0.0
            self.P[:, i] = 0.0
            self.P[i, i] = self.p0_vel_reset

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:5].copy()

    @property
    def velocity_covariance_trace(self):
        return float(self.P[3, 3] + self.P[4, 4])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m pytest test/test_kalman.py -v`
Expected: PASS (6 passed). 수치 부등식(`0.0 < kf.position[0] < 0.002` 등)이 게인 값에 따라
어긋나면 `q_pos/q_vel/r_xy/r_z` 테스트 인자를 조정해 재확인한다(로직이 아니라 파라미터 이슈이므로).

- [ ] **Step 5: 커밋**

```bash
git add src/robot_control/robot_control/kalman.py src/robot_control/test/test_kalman.py
git commit -m "feat(robot_control): add constant-velocity Kalman filter for tool tracking"
```

---

### Task 3: `servo_loop.py` — calculate 모듈 구현

**Files:**
- Modify: `src/robot_control/robot_control/servo_loop.py` (전체 교체)
- Modify: `src/robot_control/test/test_servo_loop.py` (전체 교체)

**Interfaces:**
- Consumes: `KalmanXYZV` (Task 2)
- Produces: `ServoLoop(kp_xy, kp_yaw, v_max, descend_speed, eps_descend, eps_grasp, n_stable,
  dt_latency, timeout_s, t_lost_s, innov_low=0.010, innov_high=0.040, w_alpha=0.3, z_close=0.02,
  diverge_n=5, cov_threshold=0.05)`, `.start(tool_class, grasp_width_mm, grasp_force_n)`,
  `.on_tool_track(msg)`(msg는 `header.stamp.sec/nanosec`, `pose.position.x/y/z`, `depth_valid` 속성
  필요), `.step(tcp_pose, now) -> ServoCommand`(tcp_pose는 `(x,y,z,rx,ry,rz)` 튜플, now는
  `time.monotonic()` 값), `.get_state() -> str`, `.should_close() -> bool`,
  `.should_abort() -> str|None` — Task 4(`robot_control_node.py`)에서 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`src/robot_control/test/test_servo_loop.py`:
```python
import time
import pytest
from robot_control.servo_loop import ServoLoop, ServoState, ServoCommand


class FakeHeader:
    def __init__(self, t):
        self.stamp = FakeStamp(t)


class FakeStamp:
    def __init__(self, t):
        self.sec = int(t)
        self.nanosec = int((t - int(t)) * 1e9)


class FakePosition:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakePose:
    def __init__(self, x, y, z):
        self.position = FakePosition(x, y, z)


class FakeToolTrack:
    def __init__(self, t, x, y, z, depth_valid=True):
        self.header = FakeHeader(t)
        self.pose = FakePose(x, y, z)
        self.depth_valid = depth_valid


def _make_loop(**overrides):
    kwargs = dict(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                  eps_descend=0.015, eps_grasp=0.005, n_stable=3,
                  dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                  innov_low=0.010, innov_high=0.040, w_alpha=1.0,
                  z_close=0.02, diverge_n=5, cov_threshold=0.5)
    kwargs.update(overrides)
    return ServoLoop(**kwargs)


def test_initial_state_is_tracking():
    loop = _make_loop()
    assert loop.get_state() == ServoState.TRACKING


def test_step_before_any_track_returns_zero_command():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    cmd = loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert isinstance(cmd, ServoCommand)
    assert cmd.vx == 0.0 and cmd.vy == 0.0


def test_on_tool_track_then_step_moves_toward_target():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.80, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.78, 0.0, 0.05))
    cmd = loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert cmd.vx > 0.0


def test_step_respects_v_max():
    loop = _make_loop(v_max=0.05)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 1.0, 1.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 1.0, 1.0, 0.05))
    cmd = loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    speed = (cmd.vx ** 2 + cmd.vy ** 2) ** 0.5
    assert speed <= 0.05 + 1e-9


def test_large_innovation_resets_velocity_covariance_and_drops_w():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    trace_before = loop._filter.velocity_covariance_trace
    loop.on_tool_track(FakeToolTrack(0.04, 0.5, 0.0, 0.05))
    assert loop._w == pytest.approx(0.0, abs=1e-6)
    assert loop._filter.velocity_covariance_trace >= trace_before


def test_depth_invalid_track_does_not_move_z_estimate():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05, depth_valid=True))
    loop.on_tool_track(FakeToolTrack(0.02, 0.5, 0.0, 999.0, depth_valid=False))
    assert loop._filter.position[2] == pytest.approx(0.05, abs=1e-6)


def test_should_close_requires_stable_error_and_z_gap_and_low_covariance():
    loop = _make_loop(eps_grasp=0.01, n_stable=2, z_close=0.05)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.0, 0.0, 0.05))
    loop.on_tool_track(FakeToolTrack(0.02, 0.0, 0.0, 0.05))
    for _ in range(3):
        loop.step((0.0, 0.0, 0.05, 0, 0, 0), time.monotonic())
    assert loop.should_close() is True
    assert loop.get_state() == ServoState.CLOSING


def test_should_abort_timeout():
    loop = _make_loop(timeout_s=0.0)
    loop.start('spanner', 30.0, 20.0)
    assert loop.should_abort() == 'timeout'


def test_should_abort_tracking_lost():
    loop = _make_loop(t_lost_s=0.0)
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05))
    time.sleep(0.01)
    assert loop.should_abort() == 'tracking_lost'


def test_should_abort_none_when_healthy():
    loop = _make_loop()
    loop.start('spanner', 30.0, 20.0)
    loop.on_tool_track(FakeToolTrack(0.0, 0.5, 0.0, 0.05))
    assert loop.should_abort() is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m pytest test/test_servo_loop.py -v`
Expected: FAIL (기존 `NotImplementedError` 스텁 때문에 다수 실패)

- [ ] **Step 3: 구현**

`src/robot_control/robot_control/servo_loop.py` (전체 교체):
```python
import time

import numpy as np

from robot_control.kalman import KalmanXYZV


class ServoState:
    TRACKING = 'tracking'
    DESCENDING = 'descending'
    CLOSING = 'closing'
    LIFTING = 'lifting'


class ServoCommand:
    def __init__(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.yaw_rate = yaw_rate


class ServoLoop:
    """robot_control 내부 PBVS 서보 루프 + calculate 모듈 (전체 계획.md 2절)."""

    def __init__(self, kp_xy, kp_yaw, v_max, descend_speed,
                 eps_descend, eps_grasp, n_stable, dt_latency,
                 timeout_s, t_lost_s,
                 innov_low=0.010, innov_high=0.040, w_alpha=0.3,
                 z_close=0.02, diverge_n=5, cov_threshold=0.05):
        self.kp_xy = kp_xy
        self.kp_yaw = kp_yaw
        self.v_max = v_max
        self.descend_speed = descend_speed
        self.eps_descend = eps_descend
        self.eps_grasp = eps_grasp
        self.n_stable = n_stable
        # dt_latency에 2.3절의 "루프 반주기" 보정도 함께 흡수한다(1차 구현 단순화).
        self.dt_latency = dt_latency
        self.timeout_s = timeout_s
        self.t_lost_s = t_lost_s
        self.innov_low = innov_low
        self.innov_high = innov_high
        self.w_alpha = w_alpha
        self.z_close = z_close
        self.diverge_n = diverge_n
        self.cov_threshold = cov_threshold

        self._state = ServoState.TRACKING
        self._filter = KalmanXYZV()
        self._w = 0.0
        self._last_track_time = None
        self._last_msg_time = None
        self._start_time = None
        self._stable_count = 0
        self._error_history = []
        self._last_z_gap = None

    def start(self, tool_class, grasp_width_mm, grasp_force_n):
        self.tool_class = tool_class
        self.grasp_width_mm = grasp_width_mm
        self.grasp_force_n = grasp_force_n
        self._state = ServoState.TRACKING
        self._filter = KalmanXYZV()
        self._w = 0.0
        self._last_track_time = None
        self._last_msg_time = None
        self._start_time = time.monotonic()
        self._stable_count = 0
        self._error_history = []
        self._last_z_gap = None

    def on_tool_track(self, msg):
        now = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.position

        if not self._filter._initialized:
            self._filter.initialize(pos.x, pos.y, pos.z)
            self._last_track_time = stamp
            self._last_msg_time = now
            return

        dt = max(stamp - self._last_track_time, 1e-3)
        self._filter.predict(dt)

        if msg.depth_valid:
            innov_xy = self._filter.update_xyz([pos.x, pos.y, pos.z])
        else:
            innov_xy = self._filter.update_xy_only([pos.x, pos.y])

        if innov_xy >= self.innov_high:
            self._filter.reset_velocity_covariance()
            w_target = 0.0
        elif innov_xy <= self.innov_low:
            w_target = 1.0
        else:
            span = self.innov_high - self.innov_low
            w_target = 1.0 - (innov_xy - self.innov_low) / span

        self._w = self.w_alpha * w_target + (1.0 - self.w_alpha) * self._w
        self._last_track_time = stamp
        self._last_msg_time = now

    def step(self, tcp_pose, now):
        if not self._filter._initialized:
            return ServoCommand()

        p_ref = self._filter.predict_position(self.dt_latency)
        v_tool = self._filter.velocity

        tcp_x, tcp_y, tcp_z = tcp_pose[0], tcp_pose[1], tcp_pose[2]
        e_x = p_ref[0] - tcp_x
        e_y = p_ref[1] - tcp_y
        e_xy_norm = float(np.hypot(e_x, e_y))

        self._error_history.append(e_xy_norm)
        if len(self._error_history) > self.diverge_n:
            self._error_history.pop(0)

        self._last_z_gap = abs(tcp_z - p_ref[2])

        vx = self._w * v_tool[0] + self.kp_xy * e_x
        vy = self._w * v_tool[1] + self.kp_xy * e_y
        speed = float(np.hypot(vx, vy))
        if speed > self.v_max:
            scale = self.v_max / speed
            vx *= scale
            vy *= scale

        if e_xy_norm < self.eps_descend:
            self._state = ServoState.DESCENDING
            vz = -self.descend_speed
        else:
            self._state = ServoState.TRACKING
            vz = 0.0

        if e_xy_norm < self.eps_grasp:
            self._stable_count += 1
        else:
            self._stable_count = 0

        return ServoCommand(vx=vx, vy=vy, vz=vz, yaw_rate=0.0)

    def get_state(self):
        return self._state

    def should_close(self):
        if self._stable_count < self.n_stable:
            return False
        if self._last_z_gap is None or self._last_z_gap >= self.z_close:
            return False
        if self._filter.velocity_covariance_trace >= self.cov_threshold:
            return False
        self._state = ServoState.CLOSING
        return True

    def should_abort(self):
        if self._start_time is not None and time.monotonic() - self._start_time > self.timeout_s:
            return 'timeout'
        if self._last_msg_time is not None and time.monotonic() - self._last_msg_time > self.t_lost_s:
            return 'tracking_lost'
        if len(self._error_history) == self.diverge_n and all(
                self._error_history[i] < self._error_history[i + 1]
                for i in range(len(self._error_history) - 1)):
            return 'diverging'
        # 참고: "공구가 방향 전환하여 시야 이탈 예상"(2.8절)은 판정 기준이 모호해
        # 과설계 우려가 있으므로 1차 구현 범위에서 제외했다. 실측 후 필요하면 추가한다.
        return None
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m pytest test/test_servo_loop.py -v`
Expected: PASS (전부). 실패 시 해당 테스트의 파라미터(예: `n_stable`, `z_close`, `cov_threshold`)를
로직이 아니라 수치만 조정해서 재확인.

- [ ] **Step 5: 커밋**

```bash
git add src/robot_control/robot_control/servo_loop.py src/robot_control/test/test_servo_loop.py
git commit -m "feat(robot_control): implement PBVS control law, feedforward weight, grasp/abort judgement"
```

---

### Task 4: `robot_control_node.py` — TCP pose를 `step()`에 배선

**Files:**
- Modify: `src/robot_control/robot_control/robot_control_node.py`
- Modify: `src/robot_control/test/test_robot_control_node.py`

**Interfaces:**
- Consumes: `ServoLoop.step(tcp_pose, now)` (Task 3)
- Produces: `RobotControlNode._get_current_tcp_pose() -> tuple` (스텁, 팀원1 구현 예정)

- [ ] **Step 1: 기존 테스트 중 `step` 호출부 갱신**

`src/robot_control/test/test_robot_control_node.py`에서 `node.servo_loop.step = lambda: None`
두 곳(72번째줄 부근 `test_execute_servo_pick_success_...`와 `test_dispatch_routes_servo_pick`)을 찾아
아래처럼 인자를 받도록 교체하고, 두 테스트 모두에 `node._get_current_tcp_pose = lambda: (0.0, 0.0, 0.05, 0, 0, 0)`
줄을 `node.servo_loop.start = lambda *a, **k: None` 다음 줄에 추가:
```python
    node.servo_loop.step = lambda *a, **k: None
    node._get_current_tcp_pose = lambda: (0.0, 0.0, 0.05, 0, 0, 0)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m pytest test/test_robot_control_node.py -v`
Expected: 위 두 테스트가 `TypeError: <lambda>() takes 0 positional arguments but 2 were given`로 FAIL
(아직 `_execute_servo_pick`이 `step()`을 인자 없이 호출하기 때문)

- [ ] **Step 3: `robot_control_node.py` 수정**

`src/robot_control/robot_control/robot_control_node.py`에서 `_execute_servo_pick` 위에 스텁 추가:
```python
    def _get_current_tcp_pose(self):
        """Doosan RT 세션에서 현재 TCP pose(base_link 기준 x,y,z,rx,ry,rz)를 읽는다."""
        raise NotImplementedError('_get_current_tcp_pose 구현 필요 (Doosan RT API)')
```

`_execute_servo_pick`의 루프 본문에서 `self.servo_loop.step()` 호출부를 찾아 교체:
```python
                self.servo_loop.step()
                time.sleep(0.01)
```
→
```python
                tcp_pose = self._safe_call(self._get_current_tcp_pose, default=None)
                if tcp_pose is not None:
                    self.servo_loop.step(tcp_pose, time.monotonic())
                time.sleep(0.01)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m pytest test/ -v`
Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/robot_control/robot_control/robot_control_node.py src/robot_control/test/test_robot_control_node.py
git commit -m "feat(robot_control): wire current TCP pose into servo_loop.step()"
```

---

### Task 5: `vision_node/tracking.py` — 추적·좌표변환 순수 함수

**Files:**
- Create: `src/vision_node/vision_node/tracking.py`
- Test: `src/vision_node/test/test_tracking.py`

**Interfaces:**
- Produces: `pixel_to_camera_xyz(px, py, depth, fx, fy, ppx, ppy) -> (x,y,z)`,
  `quaternion_to_rotation_matrix(x,y,z,w) -> 3x3 list`,
  `transform_to_matrix(translation, rotation) -> 4x4 list`,
  `camera_to_base(camera_xyz, tf_matrix) -> (x,y,z)`,
  `is_approaching(position_xy, velocity_xy, ref_xy) -> bool`,
  `ToolTracker(alpha=0.6, beta=0.3)` with
  `.update(detections, tool_class, reconstruct_fn, stamp) -> (position, velocity, depth_valid) | None`
  (`reconstruct_fn(cx, cy) -> (x,y,z,depth_valid) | None`), `.last_valid_z` — Task 7에서 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`src/vision_node/test/test_tracking.py`:
```python
import math
import pytest
from vision_node.tracking import (
    pixel_to_camera_xyz, quaternion_to_rotation_matrix, transform_to_matrix,
    camera_to_base, is_approaching, ToolTracker,
)


class FakeDetection:
    def __init__(self, class_name, score, x1, y1, x2, y2):
        self.class_name = class_name
        self.score = score
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


def test_pixel_to_camera_xyz_center_pixel_is_zero_xy():
    x, y, z = pixel_to_camera_xyz(320, 240, 0.5, fx=600, fy=600, ppx=320, ppy=240)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.5)


def test_quaternion_identity_is_identity_matrix():
    r = quaternion_to_rotation_matrix(0, 0, 0, 1)
    assert r[0][0] == pytest.approx(1.0)
    assert r[1][1] == pytest.approx(1.0)
    assert r[2][2] == pytest.approx(1.0)
    assert r[0][1] == pytest.approx(0.0)


def test_camera_to_base_applies_translation():
    tf_matrix = transform_to_matrix((1.0, 2.0, 3.0), (0, 0, 0, 1))
    x, y, z = camera_to_base((0.1, 0.2, 0.3), tf_matrix)
    assert (x, y, z) == pytest.approx((1.1, 2.2, 3.3))


def test_is_approaching_true_when_moving_toward_ref():
    assert is_approaching((0.5, 0.0), (0.1, 0.0), (1.0, 0.0)) is True


def test_is_approaching_false_when_moving_away():
    assert is_approaching((0.5, 0.0), (-0.1, 0.0), (1.0, 0.0)) is False


def test_tracker_returns_none_when_no_matching_class():
    tracker = ToolTracker()
    dets = [FakeDetection('hammer', 0.9, 0, 0, 10, 10)]
    result = tracker.update(dets, 'spanner', lambda cx, cy: (0, 0, 0.05, True), stamp=0.0)
    assert result is None


def test_tracker_first_frame_uses_highest_score_and_zero_velocity():
    tracker = ToolTracker()
    dets = [
        FakeDetection('spanner', 0.5, 0, 0, 10, 10),
        FakeDetection('spanner', 0.9, 20, 20, 30, 30),
    ]

    def reconstruct(cx, cy):
        return (cx / 100.0, cy / 100.0, 0.05, True)

    position, velocity, depth_valid = tracker.update(dets, 'spanner', reconstruct, stamp=0.0)
    assert position == pytest.approx((0.25, 0.25, 0.05))
    assert velocity == pytest.approx((0.0, 0.0))
    assert depth_valid is True


def test_tracker_second_frame_estimates_velocity():
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    dets1 = [FakeDetection('spanner', 0.9, 0, 0, 0, 0)]
    dets2 = [FakeDetection('spanner', 0.9, 0, 0, 0, 0)]

    tracker.update(dets1, 'spanner', lambda cx, cy: (0.0, 0.0, 0.05, True), stamp=0.0)
    position, velocity, _ = tracker.update(
        dets2, 'spanner', lambda cx, cy: (0.1, 0.0, 0.05, True), stamp=1.0)

    assert position[0] == pytest.approx(0.1, abs=1e-6)
    assert velocity[0] == pytest.approx(0.1, abs=1e-6)


def test_tracker_holds_last_valid_z_when_depth_invalid():
    tracker = ToolTracker()
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy: (0.0, 0.0, 0.05, True), stamp=0.0)
    position, _, depth_valid = tracker.update(
        [FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
        lambda cx, cy: (0.1, 0.0, 999.0, False), stamp=0.1)
    assert depth_valid is False
    assert position[2] == pytest.approx(0.05, abs=1e-6)


def test_tracker_picks_nearest_candidate_to_previous_position():
    tracker = ToolTracker()
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy: (0.0, 0.0, 0.05, True), stamp=0.0)

    dets = [
        FakeDetection('spanner', 0.9, 100, 0, 100, 0),
        FakeDetection('spanner', 0.9, 1, 0, 1, 0),
    ]

    def reconstruct(cx, cy):
        return (1.0, 0.0, 0.05, True) if cx == 100 else (0.01, 0.0, 0.05, True)

    position, _, _ = tracker.update(dets, 'spanner', reconstruct, stamp=0.1)
    assert position[0] == pytest.approx(0.01, abs=1e-3)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/vision_node && python3 -m pytest test/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision_node.tracking'`

- [ ] **Step 3: 구현**

`src/vision_node/vision_node/tracking.py`:
```python
import math


def pixel_to_camera_xyz(px, py, depth, fx, fy, ppx, ppy):
    """픽셀 좌표 + depth(camera 기준 z)를 camera 좌표계 3D 점으로 변환."""
    x = (px - ppx) * depth / fx
    y = (py - ppy) * depth / fy
    return x, y, depth


def quaternion_to_rotation_matrix(x, y, z, w):
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transform_to_matrix(translation, rotation):
    """translation=(x,y,z), rotation=(x,y,z,w) 쿼터니언 -> 4x4 변환행렬(중첩 리스트)."""
    r = quaternion_to_rotation_matrix(*rotation)
    return [
        [r[0][0], r[0][1], r[0][2], translation[0]],
        [r[1][0], r[1][1], r[1][2], translation[1]],
        [r[2][0], r[2][1], r[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def camera_to_base(camera_xyz, tf_matrix):
    x, y, z = camera_xyz
    out = []
    for row in tf_matrix[:3]:
        out.append(row[0] * x + row[1] * y + row[2] * z + row[3])
    return tuple(out)


def is_approaching(position_xy, velocity_xy, ref_xy):
    dx = ref_xy[0] - position_xy[0]
    dy = ref_xy[1] - position_xy[1]
    dot = velocity_xy[0] * dx + velocity_xy[1] * dy
    return dot > 0.0


class ToolTracker:
    """vision_node의 TRACK_TOOL 단순 추적기: 최근접 매칭 + 알파-베타 속도 필터."""

    def __init__(self, alpha=0.6, beta=0.3):
        self.alpha = alpha
        self.beta = beta
        self.position = None
        self.velocity = (0.0, 0.0)
        self.last_valid_z = None
        self.last_time = None

    def reset(self):
        self.position = None
        self.velocity = (0.0, 0.0)
        self.last_valid_z = None
        self.last_time = None

    def update(self, detections, tool_class, reconstruct_fn, stamp):
        candidates = [d for d in detections if d.class_name == tool_class]
        if not candidates:
            return None

        reconstructed = []
        for d in candidates:
            cx = (d.x1 + d.x2) / 2.0
            cy = (d.y1 + d.y2) / 2.0
            r = reconstruct_fn(cx, cy)
            if r is not None:
                reconstructed.append((r, d.score))
        if not reconstructed:
            return None

        if self.position is None:
            chosen, _ = max(reconstructed, key=lambda item: item[1])
        else:
            def dist(item):
                r = item[0]
                return math.dist((r[0], r[1], r[2]), self.position)
            chosen, _ = min(reconstructed, key=dist)

        x, y, z, depth_valid = chosen
        if depth_valid:
            self.last_valid_z = z
        elif self.last_valid_z is not None:
            z = self.last_valid_z

        return self._filter_update(x, y, z, depth_valid, stamp)

    def _filter_update(self, x, y, z, depth_valid, stamp):
        if self.position is None or self.last_time is None:
            self.position = (x, y, z)
            self.velocity = (0.0, 0.0)
            self.last_time = stamp
            return self.position, self.velocity, depth_valid

        dt = max(stamp - self.last_time, 1e-3)
        raw_vx = (x - self.position[0]) / dt
        raw_vy = (y - self.position[1]) / dt

        smoothed_x = self.position[0] + self.alpha * (x - self.position[0])
        smoothed_y = self.position[1] + self.alpha * (y - self.position[1])
        vx = self.velocity[0] + self.beta * (raw_vx - self.velocity[0])
        vy = self.velocity[1] + self.beta * (raw_vy - self.velocity[1])

        self.position = (smoothed_x, smoothed_y, z)
        self.velocity = (vx, vy)
        self.last_time = stamp
        return self.position, self.velocity, depth_valid
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/vision_node && python3 -m pytest test/test_tracking.py -v`
Expected: PASS (전부)

- [ ] **Step 5: 커밋**

```bash
git add src/vision_node/vision_node/tracking.py src/vision_node/test/test_tracking.py
git commit -m "feat(vision_node): add nearest-match tool tracker and coordinate transform helpers"
```

---

### Task 6: `vision_node/hand_tracking.py` — MediaPipe 래퍼

**Files:**
- Create: `src/vision_node/vision_node/hand_tracking.py`
- Test: `src/vision_node/test/test_hand_tracking.py`

**Interfaces:**
- Produces: `create_hands_detector()`(mediapipe 지연 임포트), `detect_hand_wrist_pixel(hands_detector,
  bgr_image) -> (px, py) | None` — Task 7에서 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`src/vision_node/test/test_hand_tracking.py`:
```python
import numpy as np
from vision_node.hand_tracking import detect_hand_wrist_pixel


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class FakeResult:
    def __init__(self, multi_hand_landmarks):
        self.multi_hand_landmarks = multi_hand_landmarks


class FakeHandsDetector:
    def __init__(self, result):
        self._result = result

    def process(self, rgb_image):
        return self._result


def test_returns_none_when_no_hand_detected():
    detector = FakeHandsDetector(FakeResult(None))
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert detect_hand_wrist_pixel(detector, image) is None


def test_returns_wrist_pixel_scaled_by_image_size():
    landmarks = [FakeLandmark(0.5, 0.25)] + [FakeLandmark(0.0, 0.0)] * 20
    detector = FakeHandsDetector(FakeResult([FakeHandLandmarks(landmarks)]))
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200

    px, py = detect_hand_wrist_pixel(detector, image)

    assert px == 100
    assert py == 25
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/vision_node && python3 -m pytest test/test_hand_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision_node.hand_tracking'`

- [ ] **Step 3: 구현**

`src/vision_node/vision_node/hand_tracking.py`:
```python
"""MediaPipe 손목 픽셀 검출. 얇은 래퍼 — 모델 로딩과 랜드마크 인덱스만 감싼다."""

import cv2


def create_hands_detector():
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


def detect_hand_wrist_pixel(hands_detector, bgr_image):
    """bgr_image(np.ndarray)에서 손목(WRIST, landmark index 0) 픽셀 좌표 (px, py)를 반환.
    검출 실패 시 None."""
    h, w = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_image)
    if not result.multi_hand_landmarks:
        return None
    wrist = result.multi_hand_landmarks[0].landmark[0]
    return int(wrist.x * w), int(wrist.y * h)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/vision_node && python3 -m pytest test/test_hand_tracking.py -v`
Expected: PASS (2개 모두). `mediapipe`가 설치돼 있지 않아도 이 테스트는 `create_hands_detector`를
호출하지 않으므로 통과한다(지연 임포트).

- [ ] **Step 5: 커밋**

```bash
git add src/vision_node/vision_node/hand_tracking.py src/vision_node/test/test_hand_tracking.py
git commit -m "feat(vision_node): add MediaPipe wrist-pixel detection wrapper"
```

---

### Task 7: `vision_node.py` — 검출 토픽 배선 + `_track_tool`/`_track_hand` 구현

**Files:**
- Modify: `src/vision_node/vision_node/vision_node.py` (전체 교체)
- Modify: `src/vision_node/test/test_vision_node.py` (전체 교체)
- Modify: `src/vision_node/package.xml`

**Interfaces:**
- Consumes: `tracking.ToolTracker`, `tracking.pixel_to_camera_xyz/transform_to_matrix/camera_to_base/
  is_approaching` (Task 5), `hand_tracking.create_hands_detector/detect_hand_wrist_pixel` (Task 6),
  `handover_interfaces.msg.DetectionArray` (Task 1)

- [ ] **Step 1: `package.xml`에 의존성 추가**

`src/vision_node/package.xml`의 `<depend>message_filters</depend>` 다음 줄에 추가:
```xml
  <depend>cv_bridge</depend>
```
파일 맨 아래 `<export>` 앞에 주석으로 pip 의존성 명시:
```xml
  <!-- mediapipe는 rosdep 대상이 아니므로 pip로 별도 설치: pip install mediapipe -->
```

- [ ] **Step 2: 실패하는 테스트 작성**

`src/vision_node/test/test_vision_node.py` (전체 교체):
```python
import rclpy
import pytest

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from handover_interfaces.msg import ToolTrack, DetectionArray, Detection2D
from handover_interfaces.srv import SetVisionMode
from vision_node.vision_node import VisionNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = VisionNode()
    yield n
    n.destroy_node()


def _make_image_msg():
    msg = Image()
    msg.header = Header()
    msg.header.frame_id = 'camera_link'
    return msg


def _make_info_msg():
    msg = CameraInfo()
    msg.k = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
    return msg


def _make_detection_msg(detections=None):
    msg = DetectionArray()
    msg.detections = detections or []
    return msg


def test_set_mode_updates_state(node):
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_TOOL
    request.tool_class = 'spanner'
    response = SetVisionMode.Response()

    result = node._on_set_mode(request, response)

    assert result.success is True
    assert node.mode == SetVisionMode.Request.TRACK_TOOL
    assert node.tool_class == 'spanner'


def test_set_mode_off(node):
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.OFF
    request.tool_class = ''
    response = SetVisionMode.Response()

    result = node._on_set_mode(request, response)

    assert result.success is True
    assert node.mode == SetVisionMode.Request.OFF


def test_synced_images_dispatches_to_track_tool_and_publishes(node):
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tool_class = 'spanner'
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'

    expected_track = ToolTrack()
    expected_track.tool_class = 'spanner'
    node._track_tool = lambda color, depth, info, detection, tf, tool_class: expected_track

    published = []
    node.pub_tool_track.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert published == [expected_track]


def test_synced_images_skips_publish_when_track_tool_returns_none(node):
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._track_tool = lambda *a, **k: None

    published = []
    node.pub_tool_track.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert published == []


def test_synced_images_skips_when_tf_lookup_fails(node):
    from tf2_ros import TransformException

    def _raise(*a, **k):
        raise TransformException('no tf yet')

    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = _raise

    called = []
    node._track_tool = lambda *a, **k: called.append(1)

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert called == []


def test_synced_images_dispatches_to_track_hand(node):
    node.mode = SetVisionMode.Request.TRACK_HAND
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'

    from geometry_msgs.msg import PoseStamped
    expected_pose = PoseStamped()
    node._track_hand = lambda color, depth, info, tf: expected_pose

    published = []
    node.pub_hand_pose.publish = published.append

    node._on_synced_images(
        _make_image_msg(), _make_image_msg(), _make_info_msg(), _make_detection_msg())

    assert published == [expected_pose]


def test_track_tool_filters_by_class_and_reconstructs_position(node):
    node.tf_buffer.lookup_transform = lambda *a, **k: None

    class FakeTransform:
        class transform:
            class translation:
                x = 0.0
                y = 0.0
                z = 0.0
            class rotation:
                x = 0.0
                y = 0.0
                z = 0.0
                w = 1.0

    color_msg = _make_image_msg()
    depth_msg = _make_image_msg()
    info_msg = _make_info_msg()
    detection = Detection2D()
    detection.class_name = 'spanner'
    detection.score = 0.9
    detection.x1, detection.y1, detection.x2, detection.y2 = 310, 230, 330, 250
    detection_msg = _make_detection_msg([detection])

    import numpy as np
    fake_depth = np.full((480, 424), 500, dtype=np.uint16)  # 0.5m
    node._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding=None: fake_depth

    track = node._track_tool(
        color_msg, depth_msg, info_msg, detection_msg, FakeTransform(), 'spanner')

    assert track is not None
    assert track.tool_class == 'spanner'
    assert track.pose.position.z == pytest.approx(0.5, abs=1e-3)
    assert track.depth_valid is True
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/vision_node && python3 -m pytest test/test_vision_node.py -v`
Expected: FAIL (시그니처 불일치·`NotImplementedError` 등 다수 실패)

- [ ] **Step 4: 구현**

`src/vision_node/vision_node/vision_node.py` (전체 교체):
```python
import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from handover_interfaces.msg import ToolTrack, DetectionArray
from handover_interfaces.srv import SetVisionMode

from vision_node.tracking import (
    ToolTracker, pixel_to_camera_xyz, transform_to_matrix, camera_to_base, is_approaching,
)
from vision_node.hand_tracking import create_hands_detector, detect_hand_wrist_pixel


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.mode = SetVisionMode.Request.OFF
        self.tool_class = ''

        self.declare_parameter('vision.min_z_m', 0.10)
        self.declare_parameter('vision.approach_ref_x', 0.0)
        self.declare_parameter('vision.approach_ref_y', 0.0)
        self.declare_parameter('vision.tracker_alpha', 0.6)
        self.declare_parameter('vision.tracker_beta', 0.3)
        self.min_z_m = self.get_parameter('vision.min_z_m').value
        self.approach_ref_xy = (
            self.get_parameter('vision.approach_ref_x').value,
            self.get_parameter('vision.approach_ref_y').value,
        )

        self._bridge = CvBridge()
        self.tracker = ToolTracker(
            alpha=self.get_parameter('vision.tracker_alpha').value,
            beta=self.get_parameter('vision.tracker_beta').value)
        self._hands_detector = None  # 지연 생성 (TRACK_HAND 최초 진입 시)

        self.pub_tool_track = self.create_publisher(ToolTrack, '/vision/tool_track', 10)
        self.pub_hand_pose = self.create_publisher(PoseStamped, '/vision/hand_pose', 10)
        self.srv_set_mode = self.create_service(SetVisionMode, '/vision/set_mode', self._on_set_mode)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_color = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self.sub_detections = message_filters.Subscriber(
            self, DetectionArray, '/detection/tool_boxes')
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info, self.sub_detections],
            queue_size=10, slop=0.05)
        self._sync.registerCallback(self._on_synced_images)

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _on_set_mode(self, request, response):
        self.mode = request.mode
        self.tool_class = request.tool_class
        if request.mode == SetVisionMode.Request.TRACK_TOOL:
            self.tracker.reset()
        response.success = True
        response.message = f'mode set to {request.mode} (tool_class={request.tool_class})'
        return response

    def _on_synced_images(self, color_msg, depth_msg, info_msg, detection_msg):
        try:
            tf_at_stamp = self.tf_buffer.lookup_transform(
                'base_link', color_msg.header.frame_id, color_msg.header.stamp,
                timeout=Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup failed: {ex}')
            return

        if self.mode == SetVisionMode.Request.TRACK_TOOL:
            track = self._safe_call(
                self._track_tool, color_msg, depth_msg, info_msg, detection_msg,
                tf_at_stamp, self.tool_class, default=None)
            if track is not None:
                self.pub_tool_track.publish(track)
        elif self.mode == SetVisionMode.Request.TRACK_HAND:
            hand_pose = self._safe_call(
                self._track_hand, color_msg, depth_msg, info_msg, tf_at_stamp, default=None)
            if hand_pose is not None:
                self.pub_hand_pose.publish(hand_pose)

    def _tf_matrix(self, tf_at_stamp):
        t = tf_at_stamp.transform.translation
        r = tf_at_stamp.transform.rotation
        return transform_to_matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    def _track_tool(self, color_msg, depth_msg, info_msg, detection_msg, tf_at_stamp, tool_class):
        """저해상도 검출(팀원3 제공) + 3D 복원(tf_at_stamp 사용) + 알파-베타 필터로 ToolTrack을 만든다."""
        fx, fy, ppx, ppy = info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5]
        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        tf_matrix = self._tf_matrix(tf_at_stamp)

        def reconstruct(cx, cy):
            px, py = int(cx), int(cy)
            if not (0 <= py < depth_image.shape[0] and 0 <= px < depth_image.shape[1]):
                return None
            depth_m = float(depth_image[py, px]) / 1000.0
            depth_valid = depth_m >= self.min_z_m
            z = depth_m if depth_valid else (self.tracker.last_valid_z or 0.0)
            cam_xyz = pixel_to_camera_xyz(px, py, z, fx, fy, ppx, ppy)
            base_xyz = camera_to_base(cam_xyz, tf_matrix)
            return (base_xyz[0], base_xyz[1], base_xyz[2], depth_valid)

        stamp = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
        result = self.tracker.update(detection_msg.detections, tool_class, reconstruct, stamp)
        if result is None:
            return None

        position, velocity, depth_valid = result
        track = ToolTrack()
        track.header = color_msg.header
        track.tool_class = tool_class
        track.pose.position.x = position[0]
        track.pose.position.y = position[1]
        track.pose.position.z = position[2]
        track.pose.orientation.w = 1.0
        track.velocity.x = velocity[0]
        track.velocity.y = velocity[1]
        track.velocity.z = 0.0
        track.depth_valid = depth_valid
        track.approaching = is_approaching(
            (position[0], position[1]), velocity, self.approach_ref_xy)
        track.confidence = 1.0
        return track

    def _track_hand(self, color_msg, depth_msg, info_msg, tf_at_stamp):
        """MediaPipe로 손목을 검출해 PoseStamped를 만든다."""
        if self._hands_detector is None:
            self._hands_detector = create_hands_detector()

        image = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        wrist_px = detect_hand_wrist_pixel(self._hands_detector, image)
        if wrist_px is None:
            return None

        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        px, py = wrist_px
        if not (0 <= py < depth_image.shape[0] and 0 <= px < depth_image.shape[1]):
            return None
        depth_m = float(depth_image[py, px]) / 1000.0
        if depth_m <= 0.0:
            return None

        fx, fy, ppx, ppy = info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5]
        cam_xyz = pixel_to_camera_xyz(px, py, depth_m, fx, fy, ppx, ppy)
        base_xyz = camera_to_base(cam_xyz, self._tf_matrix(tf_at_stamp))

        pose = PoseStamped()
        pose.header = color_msg.header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = base_xyz
        pose.pose.orientation.w = 1.0
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: 테스트 통과 확인**

Run:
```bash
cd /home/hwangjeongui/rokey_proj_02
colcon build --packages-select handover_interfaces vision_node
source install/setup.bash
cd src/vision_node && python3 -m pytest test/ -v
```
Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add src/vision_node/vision_node/vision_node.py src/vision_node/test/test_vision_node.py src/vision_node/package.xml
git commit -m "feat(vision_node): wire detection topic and implement tool/hand tracking"
```

---

### Task 8: 검출 하네스 + 서보 시뮬레이션 하네스 (하드웨어 없이 튜닝)

**Files:**
- Create: `src/vision_node/vision_node/tools/__init__.py`
- Create: `src/vision_node/vision_node/tools/fake_detection_publisher.py`
- Modify: `src/vision_node/setup.py`
- Create: `src/robot_control/robot_control/tools/__init__.py`
- Create: `src/robot_control/robot_control/tools/simulate_conveyor.py`
- Test: `src/robot_control/test/test_simulate_conveyor.py`

**Interfaces:**
- Produces: `simulate_conveyor.make_scenario(name, duration_s=5.0, dt=0.02) -> list[(t,x,y,z)]`
  (name은 `'constant'|'long_reversal'|'short_oscillation'`), `simulate_conveyor.run_servo_sim(loop,
  scenario, dt=0.02) -> list[dict]`(각 step의 `w`, `e_xy`, `tcp_xy` 기록) — 사람이 직접 실행해서
  게인·`innov_low/high`·`w_alpha`를 눈으로 튜닝하는 용도

- [ ] **Step 1: `fake_detection_publisher.py` 작성 (검출 인터페이스 하네스)**

`src/vision_node/vision_node/tools/__init__.py`: (빈 파일)

`src/vision_node/vision_node/tools/fake_detection_publisher.py`:
```python
"""팀원3의 object_detection 노드가 아직 없을 때, /detection/tool_boxes에
가짜 bbox를 흘려보내 vision_node를 독립적으로 검증하기 위한 개발용 노드."""

import rclpy
from rclpy.node import Node

from handover_interfaces.msg import DetectionArray, Detection2D


class FakeDetectionPublisher(Node):
    def __init__(self):
        super().__init__('fake_detection_publisher')
        self.declare_parameter('tool_class', 'spanner')
        self.declare_parameter('rate_hz', 30.0)
        self.tool_class = self.get_parameter('tool_class').value
        rate_hz = self.get_parameter('rate_hz').value

        self.pub = self.create_publisher(DetectionArray, '/detection/tool_boxes', 10)
        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        self._t = 0.0
        self._dt = 1.0 / rate_hz

    def _on_timer(self):
        # 화면 중앙 부근에서 좌우로 왕복하는 고정 크기 bbox
        cx = 212 + 100.0 * ((self._t % 4.0) / 2.0 - 1.0)
        msg = DetectionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        det = Detection2D()
        det.class_name = self.tool_class
        det.score = 0.95
        det.x1, det.y1 = int(cx - 15), 110
        det.x2, det.y2 = int(cx + 15), 130
        msg.detections = [det]
        self.pub.publish(msg)
        self._t += self._dt


def main(args=None):
    rclpy.init(args=args)
    node = FakeDetectionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

`src/vision_node/setup.py`의 `entry_points`에 추가:
```python
    entry_points={
        'console_scripts': [
            'vision_node = vision_node.vision_node:main',
            'fake_detection_publisher = vision_node.tools.fake_detection_publisher:main',
        ],
    },
```

- [ ] **Step 2: `simulate_conveyor.py`의 실패하는 테스트 작성**

`src/robot_control/test/test_simulate_conveyor.py`:
```python
import pytest
from robot_control.servo_loop import ServoLoop
from robot_control.tools.simulate_conveyor import make_scenario, run_servo_sim


def _make_loop():
    return ServoLoop(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                      eps_descend=0.015, eps_grasp=0.005, n_stable=5,
                      dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                      innov_low=0.010, innov_high=0.040, w_alpha=0.3)


def test_constant_scenario_keeps_w_near_one():
    loop = _make_loop()
    scenario = make_scenario('constant', duration_s=3.0, dt=0.02)
    log = run_servo_sim(loop, scenario, dt=0.02)
    tail_w = [row['w'] for row in log[-20:]]
    assert sum(tail_w) / len(tail_w) > 0.8


def test_long_reversal_scenario_drops_w_after_direction_change():
    loop = _make_loop()
    scenario = make_scenario('long_reversal', duration_s=4.0, dt=0.02)
    log = run_servo_sim(loop, scenario, dt=0.02)
    min_w_after_reversal = min(row['w'] for row in log[len(log) // 2:len(log) // 2 + 25])
    assert min_w_after_reversal < 0.5


def test_short_oscillation_scenario_produces_lower_average_w_than_constant():
    loop_const = _make_loop()
    loop_osc = _make_loop()
    const_log = run_servo_sim(loop_const, make_scenario('constant', 3.0, 0.02), dt=0.02)
    osc_log = run_servo_sim(loop_osc, make_scenario('short_oscillation', 3.0, 0.02), dt=0.02)

    avg_w_const = sum(row['w'] for row in const_log) / len(const_log)
    avg_w_osc = sum(row['w'] for row in osc_log) / len(osc_log)
    assert avg_w_osc < avg_w_const
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m pytest test/test_simulate_conveyor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robot_control.tools.simulate_conveyor'`

- [ ] **Step 4: 구현**

`src/robot_control/robot_control/tools/__init__.py`: (빈 파일)

`src/robot_control/robot_control/tools/simulate_conveyor.py`:
```python
"""하드웨어 없이 ServoLoop를 튜닝하기 위한 오프라인 시뮬레이션.
컨베이어 3가지 시나리오(전체 계획.md 7절 4번)를 흉내낸 ToolTrack을 만들어 흘려보내고,
받은 v_cmd를 적분해 tcp_pose를 갱신하며 w/오차를 기록한다."""

import math


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


class _FakeToolTrack:
    def __init__(self, t, x, y, z, depth_valid=True):
        self.header = _Header(t)
        self.pose = _Pose(x, y, z)
        self.depth_valid = depth_valid


def make_scenario(name, duration_s=5.0, dt=0.02):
    """(t, x, y, z) 리스트. y=0, z=0.05 고정, x만 시나리오에 따라 변화(전부 base_link 기준[m])."""
    n = int(duration_s / dt)
    points = []
    if name == 'constant':
        v = 0.10
        for i in range(n):
            t = i * dt
            points.append((t, 0.9 - v * t, 0.0, 0.05))
    elif name == 'long_reversal':
        v = 0.10
        half = duration_s / 2.0
        for i in range(n):
            t = i * dt
            if t < half:
                x = 0.9 - v * t
            else:
                x = (0.9 - v * half) + v * (t - half)
            points.append((t, x, 0.0, 0.05))
    elif name == 'short_oscillation':
        amplitude = 0.05
        period_s = 0.5
        for i in range(n):
            t = i * dt
            x = 0.7 + amplitude * math.sin(2 * math.pi * t / period_s)
            points.append((t, x, 0.0, 0.05))
    else:
        raise ValueError(f'unknown scenario: {name}')
    return points


def run_servo_sim(loop, scenario, dt=0.02):
    """scenario를 재생하며 매 스텝 ServoLoop를 갱신하고 로그를 반환한다.
    log 각 원소: {'t','w','e_xy','tcp_x','tcp_y'}."""
    tcp_x, tcp_y, tcp_z = scenario[0][1] + 0.3, 0.0, 0.05
    log = []

    loop.start('spanner', 30.0, 20.0)

    for t, x, y, z in scenario:
        loop.on_tool_track(_FakeToolTrack(t, x, y, z))
        cmd = loop.step((tcp_x, tcp_y, tcp_z, 0, 0, 0), t)
        tcp_x += cmd.vx * dt
        tcp_y += cmd.vy * dt
        e_xy = math.hypot(x - tcp_x, y - tcp_y)
        log.append({'t': t, 'w': loop._w, 'e_xy': e_xy, 'tcp_x': tcp_x, 'tcp_y': tcp_y})

    return log


if __name__ == '__main__':
    from robot_control.servo_loop import ServoLoop

    for name in ('constant', 'long_reversal', 'short_oscillation'):
        loop = ServoLoop(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                          eps_descend=0.015, eps_grasp=0.005, n_stable=5,
                          dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3,
                          innov_low=0.010, innov_high=0.040, w_alpha=0.3)
        log = run_servo_sim(loop, make_scenario(name, duration_s=4.0, dt=0.02), dt=0.02)
        avg_w = sum(row['w'] for row in log) / len(log)
        avg_e = sum(row['e_xy'] for row in log) / len(log)
        print(f'{name}: avg_w={avg_w:.3f} avg_e_xy={avg_e:.4f}m')
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m pytest test/test_simulate_conveyor.py -v`
Expected: PASS. 통과하지 않으면 시나리오 진폭/주기나 `w_alpha` 등 숫자만 조정(로직 변경 아님).

수동 확인(선택):
```bash
cd /home/hwangjeongui/rokey_proj_02/src/robot_control && python3 -m robot_control.tools.simulate_conveyor
```
Expected: 3개 시나리오의 `avg_w`, `avg_e_xy`가 출력됨 — `constant`의 avg_w가 가장 높고
`short_oscillation`의 avg_w가 가장 낮아야 함

- [ ] **Step 6: 커밋**

```bash
git add src/vision_node/vision_node/tools/ src/vision_node/setup.py src/robot_control/robot_control/tools/ src/robot_control/test/test_simulate_conveyor.py
git commit -m "feat: add offline detection/servo simulation harnesses for hardware-free tuning"
```

---

### Task 9: 전체 빌드·테스트 통합 확인

**Files:** 없음 (검증만)

- [ ] **Step 1: 전체 빌드**

```bash
cd /home/hwangjeongui/rokey_proj_02
colcon build --packages-select handover_interfaces vision_node robot_control
```
Expected: 에러 없이 빌드 완료

- [ ] **Step 2: 전체 테스트**

```bash
source install/setup.bash
cd src/vision_node && python3 -m pytest test/ -v && cd ../robot_control && python3 -m pytest test/ -v
```
Expected: 두 패키지 전부 PASS

- [ ] **Step 3: 남은 gap 정리(코드 변경 없음, 확인만)**

다음은 이번 계획 범위 밖이며 별도로 처리해야 함을 기록만 해둔다:
- `src/vision_node/launch/vision_node.launch.py`의 hand-eye 캘리브레이션 TF 값이 아직 `0 0 0 0 0 0`
  placeholder — 실제 캘리브레이션 결과값으로 교체 필요
- `mediapipe`가 개발 환경에 설치돼 있지 않음 — `TRACK_HAND` 실제 구동 전 `pip install mediapipe` 필요
- 팀원3의 `/detection/tool_boxes` 실제 퍼블리셔가 아직 없음 — 지금은 Task 8의
  `fake_detection_publisher`로 대체 검증
- 팀원1 담당 스텁(`_get_current_tcp_pose`, `_call_move_service`, `_open_rt_session` 등)은
  이 계획에서 의도적으로 그대로 둠
