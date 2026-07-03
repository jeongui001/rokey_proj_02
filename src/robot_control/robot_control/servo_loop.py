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
    """robot_control 내부 PBVS 서보 루프 (데모.md 2절)."""

    def __init__(self, kp_xy, kp_yaw, v_max, descend_speed,
                 eps_descend, eps_grasp, n_stable, dt_latency,
                 timeout_s, t_lost_s):
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
        self._state = ServoState.TRACKING

    def start(self, tool_class: str, grasp_width_mm: float, grasp_force_n: float) -> None:
        """servo_pick goal 시작 시 호출. 필터·타이머 초기화 등."""
        raise NotImplementedError('ServoLoop.start 구현 필요')

    def on_tool_track(self, msg) -> None:
        """/vision/tool_track 수신마다 호출. 필터(칼만/알파-베타) 갱신."""
        raise NotImplementedError('ServoLoop.on_tool_track 구현 필요')

    def step(self):
        """RT 명령 주기마다 호출. PBVS 제어식(2.3절)으로 다음 명령을 계산."""
        raise NotImplementedError('ServoLoop.step 구현 필요')

    def get_state(self) -> str:
        """현재 서보 상태(tracking/descending/closing/lifting)."""
        return self._state

    def should_close(self) -> bool:
        """폐합 판정(2.6절: |e_xy|<eps_grasp가 n_stable주기 연속 ∧ z_gap<z_close ∧ 공분산<임계)."""
        raise NotImplementedError('ServoLoop.should_close 구현 필요')

    def should_abort(self):
        """발산/유실/이탈/타임아웃(2.8절) 판정. 사유 문자열 또는 None."""
        raise NotImplementedError('ServoLoop.should_abort 구현 필요')
