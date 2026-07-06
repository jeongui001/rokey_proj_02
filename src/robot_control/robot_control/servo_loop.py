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

    좌표 계약(고정): 이 클래스에 들어오는 msg.pose.position은 반드시
    "target(물체 위치) - current(TCP 위치)" 오차(base_link 프레임, m)여야 한다.
    이 변환은 robot_control_node._compute_tool_track_tcp_offset()이 GetCurrentPosx로
    조회한 현재 TCP 위치를 이용해 미리 계산해서 넘긴다 - ServoLoop 자신은 좌표 변환을
    하지 않는다. 오차가 양수이면(목표가 현재보다 +축 방향으로 더 멀면) 그 축으로
    이동해 오차를 줄이는 방향으로 속도를 낸다(부호가 같은 비례 제어: v = +Kp*error).

    orientation(TODO, 확정 필요): ToolTrack.orientation이 "목표 TCP 자세"를 의미하는지
    아직 합의되지 않았고, 절대 orientation을 상대 회전 오차처럼 임의로 계산해서도 안
    된다. 그래서 기본값(enable_yaw_control=False)에서는 yaw_rate를 항상 0으로
    고정한다 - 의미가 확정된 뒤에만 켤 것.
    """

    def __init__(self, kp_xy, kp_yaw, v_max, descend_speed,
                 eps_descend, eps_grasp, n_stable, dt_latency,
                 timeout_s, t_lost_s, z_close_m=0.01,
                 diverge_factor=1.2, diverge_window=3,
                 enable_yaw_control=False):
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
        # TODO(확정 필요): ToolTrack.orientation 의미(목표 TCP 자세인지 여부)가
        # 합의되기 전까지는 기본 false로 잠가 yaw_rate=0을 강제한다.
        self.enable_yaw_control = enable_yaw_control
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
        # error_z는 부호 있는 오차(target_z - current_z)다 - "남은 거리"처럼 항상
        # 양수라고 가정하지 않고 절댓값으로 근접 여부를 판정한다.
        error_z = msg.pose.position.z
        if error_xy < self.eps_descend:
            self._state = (
                ServoState.CLOSING if abs(error_z) <= self.z_close_m else ServoState.DESCENDING)
        else:
            self._state = ServoState.TRACKING

    def step(self):
        """RT 명령 주기마다 호출. 단순 P 제어로 다음 속도 명령을 계산한다.

        입력 position은 (target - current) 오차이므로, 오차를 줄이는 방향(오차와
        같은 부호)으로 속도를 낸다: v = +Kp * error."""
        if self._latest_msg is None:
            return ServoCommand()
        pos = self._latest_msg.pose.position
        error_x = pos.x
        error_y = pos.y
        error_z = pos.z

        vx = _clip(self.kp_xy * error_x, self.v_max)
        vy = _clip(self.kp_xy * error_y, self.v_max)

        error_xy = math.hypot(error_x, error_y)
        if error_xy < self.eps_descend and abs(error_z) > self.z_close_m:
            vz = _clip(self.kp_xy * error_z, self.descend_speed)
        else:
            vz = 0.0

        yaw_rate = 0.0
        if self.enable_yaw_control:
            # TODO(확정 필요): orientation이 목표 TCP 자세로 확정된 뒤에만 켤 것.
            yaw_error = _yaw_from_quaternion(self._latest_msg.pose.orientation)
            yaw_rate = _clip(self.kp_yaw * yaw_error, self.v_max)

        return ServoCommand(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    def get_state(self) -> str:
        return self._state

    def should_close(self) -> bool:
        if self._latest_msg is None:
            return False
        error_z = self._latest_msg.pose.position.z
        return self._stable_count >= self.n_stable and abs(error_z) <= self.z_close_m

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


class HandApproachState:
    TRACKING = 'tracking'
    ARRIVED = 'arrived'


class HandApproachServo:
    """작업자 손에 접근하는 PBVS 서보 루프.

    ServoLoop과 같은 P 제어(부호 계약: v = +Kp*error) 패턴을 재사용하지만 목표가
    "손"이고, 정지 조건이 xy/z 2단계가 아니라 **3D 유클리드 거리 하나**뿐이라는
    점이 다르다. 그리퍼 동작은 전혀 하지 않는다 - 정지 조건에 도달하면 속도 0으로
    멈추고, 그 다음은 (task_manager가 이어서 보내는) handover_hold가 당김을
    기다린다.

    좌표 계약은 ServoLoop와 동일하다: 입력 msg.pose.position은 반드시
    "손 위치 - 현재 TCP 위치" 오차(base_link 프레임, m)여야 한다
    (robot_control_node._compute_hand_pose_tcp_offset이 계산해서 넘긴다).

    /vision/hand_pose는 plain geometry_msgs/PoseStamped라 ToolTrack과 달리
    depth_valid/confidence 필드가 없다 - "추적 유실"은 메시지 수신 최신성
    (t_lost_s)만으로 판단한다. orientation(손이 향한 방향)도 아직 의미가 정의돼
    있지 않아 yaw 명령은 만들지 않는다(항상 0).
    """

    def __init__(self, kp_xy, v_max, timeout_s, t_lost_s, stop_distance_m,
                 diverge_factor=1.2, diverge_window=3):
        self.kp_xy = kp_xy
        self.v_max = v_max
        self.timeout_s = timeout_s
        self.t_lost_s = t_lost_s
        self.stop_distance_m = stop_distance_m
        self.diverge_factor = diverge_factor
        self.diverge_window = diverge_window
        self._state = HandApproachState.TRACKING
        self._start_time = None
        self._last_update_time = None
        self._latest_msg = None
        self._error_history = []

    def start(self) -> None:
        now = time.monotonic()
        self._state = HandApproachState.TRACKING
        self._start_time = now
        self._last_update_time = now
        self._latest_msg = None
        self._error_history = []

    def _distance(self, msg) -> float:
        pos = msg.pose.position
        return math.sqrt(pos.x ** 2 + pos.y ** 2 + pos.z ** 2)

    def on_hand_pose(self, msg) -> None:
        self._latest_msg = msg
        self._last_update_time = time.monotonic()
        distance = self._distance(msg)
        self._error_history.append(distance)
        if len(self._error_history) > self.diverge_window:
            self._error_history.pop(0)
        self._state = (
            HandApproachState.ARRIVED if distance <= self.stop_distance_m
            else HandApproachState.TRACKING)

    def step(self):
        """입력 position은 (손 위치 - 현재 TCP) 오차이므로, 오차를 줄이는 방향으로
        속도를 낸다: v = +Kp * error (ServoLoop.step()과 동일한 부호 계약). 손을
        향한 회전(yaw)은 orientation 의미가 정의돼 있지 않아 만들지 않는다."""
        if self._latest_msg is None:
            return ServoCommand()
        pos = self._latest_msg.pose.position
        vx = _clip(self.kp_xy * pos.x, self.v_max)
        vy = _clip(self.kp_xy * pos.y, self.v_max)
        vz = _clip(self.kp_xy * pos.z, self.v_max)
        return ServoCommand(vx=vx, vy=vy, vz=vz, yaw_rate=0.0)

    def get_state(self) -> str:
        return self._state

    def should_stop(self) -> bool:
        """3D 유클리드 거리가 stop_distance_m 이내면 접근을 멈춘다(그리퍼 동작 없음)."""
        if self._latest_msg is None:
            return False
        return self._distance(self._latest_msg) <= self.stop_distance_m

    def should_abort(self):
        now = time.monotonic()
        if self._start_time is not None and (now - self._start_time) > self.timeout_s:
            return 'timeout'
        if self._last_update_time is not None and (now - self._last_update_time) > self.t_lost_s:
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
