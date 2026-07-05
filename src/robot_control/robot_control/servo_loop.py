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
