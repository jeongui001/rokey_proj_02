import math
import time


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


def _clip(value, limit):
    if limit <= 0:
        return 0.0
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _yaw_from_quaternion(orientation) -> float:
    """geometry_msgs/Quaternion에서 yaw(rad)만 추출한다 (ToolTrack.msg 주석: orientation에는
    yaw만 의미 있게 반영됨을 전제로 한다)."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class ServoLoop:
    """robot_control 내부 PBVS 서보 루프 - 1차 MVP.

    단순 비례(P) 제어 + 속도 제한 + 추적 유실/타임아웃 감시로 구성한다.
    칼만 필터 등 상태 추정기는 넣지 않는다 (추후 단계에서 추가 예정).

    가정(TODO): vision_node의 ToolTrack.pose가 아직 미구현 상태(_track_tool이
    NotImplementedError 스텁)라 실제 좌표계가 확정되지 않았다. 이 구현은
    msg.pose.position.(x, y)를 "그리퍼(TCP) 기준 xy 오차", msg.pose.position.z를
    "남은 하강 거리"로 이미 정렬되어 들어온다고 가정한다. 만약 vision_node가 base_link
    절대좌표를 그대로 publish하도록 확정되면, robot_control 쪽에서 현재 TCP 위치
    (dsr_msgs2 GetCurrentPosx)를 빼는 변환을 추가해야 한다.
    """

    def __init__(self, kp_xy, kp_yaw, v_max, descend_speed,
                 eps_descend, eps_grasp, n_stable, dt_latency,
                 timeout_s, t_lost_s, z_close_m=0.01,
                 diverge_factor=1.2, diverge_window=3):
        self.kp_xy = kp_xy
        self.kp_yaw = kp_yaw
        self.v_max = v_max
        self.descend_speed = descend_speed
        self.eps_descend = eps_descend
        self.eps_grasp = eps_grasp
        self.n_stable = n_stable
        self.dt_latency = dt_latency
        self.timeout_s = timeout_s
        self.t_lost_s = t_lost_s
        self.z_close_m = z_close_m
        self.diverge_factor = diverge_factor
        self.diverge_window = diverge_window
        self._state = ServoState.TRACKING
        self._tool_class = None
        self._grasp_width_mm = 0.0
        self._grasp_force_n = 0.0
        self._start_time = None
        self._last_update_time = None
        self._latest_msg = None
        self._stable_count = 0
        self._error_history = []

    def start(self, tool_class: str, grasp_width_mm: float, grasp_force_n: float) -> None:
        now = time.monotonic()
        self._tool_class = tool_class
        self._grasp_width_mm = grasp_width_mm
        self._grasp_force_n = grasp_force_n
        self._state = ServoState.TRACKING
        self._start_time = now
        self._last_update_time = now
        self._latest_msg = None
        self._stable_count = 0
        self._error_history = []

    def on_tool_track(self, msg) -> None:
        self._latest_msg = msg
        self._last_update_time = time.monotonic()
        error_xy = math.hypot(msg.pose.position.x, msg.pose.position.y)
        self._error_history.append(error_xy)
        if len(self._error_history) > self.diverge_window:
            self._error_history.pop(0)
        if error_xy < self.eps_grasp:
            self._stable_count += 1
        else:
            self._stable_count = 0
        z_gap = msg.pose.position.z
        if error_xy < self.eps_descend:
            self._state = ServoState.DESCENDING if z_gap > self.z_close_m else ServoState.CLOSING
        else:
            self._state = ServoState.TRACKING

    def step(self):
        """RT 명령 주기마다 호출. 단순 P 제어로 다음 속도 명령을 계산한다."""
        if self._latest_msg is None:
            return ServoCommand()
        pos = self._latest_msg.pose.position
        error_x = pos.x
        error_y = pos.y
        z_gap = pos.z
        yaw_error = _yaw_from_quaternion(self._latest_msg.pose.orientation)

        vx = _clip(-self.kp_xy * error_x, self.v_max)
        vy = _clip(-self.kp_xy * error_y, self.v_max)
        yaw_rate = _clip(-self.kp_yaw * yaw_error, self.v_max)

        error_xy = math.hypot(error_x, error_y)
        if error_xy < self.eps_descend and z_gap > self.z_close_m:
            vz = -self.descend_speed
        else:
            vz = 0.0

        return ServoCommand(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    def get_state(self) -> str:
        return self._state

    def should_close(self) -> bool:
        if self._latest_msg is None:
            return False
        z_gap = self._latest_msg.pose.position.z
        return self._stable_count >= self.n_stable and z_gap <= self.z_close_m

    def should_abort(self):
        now = time.monotonic()
        if self._start_time is not None and (now - self._start_time) > self.timeout_s:
            return 'timeout'
        if self._last_update_time is not None and (now - self._last_update_time) > self.t_lost_s:
            return 'lost'
        if self._latest_msg is not None and not self._latest_msg.depth_valid:
            return 'lost'
        if len(self._error_history) == self.diverge_window:
            strictly_increasing = all(
                self._error_history[i] < self._error_history[i + 1]
                for i in range(len(self._error_history) - 1)
            )
            if strictly_increasing and (
                    self._error_history[-1] > self._error_history[0] * self.diverge_factor):
                return 'diverged'
        return None
