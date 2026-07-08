import math
import time

import numpy as np

from robot_control.kalman import KalmanXYZV

class ServoState:
    TRACKING = 'tracking'
    DESCENDING = 'descending'
    CLOSING = 'closing'
    LIFTING = 'lifting'


class ServoCommand:
    """RT 명령 주기마다 robot_control_node가 실제 로봇에 흘려보낼 속도 명령.
    base_link 기준 선속도(vx,vy,vz) + yaw 각속도."""

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
    """robot_control 내부 PBVS 서보 루프 + calculate 모듈 (전체 계획.md 2절).

    팀원1(robot_control_node)이 이 클래스를 어떻게 부르는지가 곧 "인터페이스":
      1. servo_pick goal 수신 시 start() 한 번
      2. /vision/tool_track 콜백마다 on_tool_track(msg) - 칼만 필터 갱신
      3. RT 명령 주기마다 step(tcp_pose, now) - 제어 명령 계산
      4. 매 틱마다 should_abort()/should_close()로 종료 조건 확인
    내부 필터·게인·임계값은 robot_control_node가 몰라도 되게 캡슐화되어 있다.
    """

    def __init__(self, kp_xy, kp_yaw, v_max, descend_speed,
                 eps_descend, eps_grasp, n_stable, dt_latency,
                 timeout_s, t_lost_s,
                 innov_low=0.010, innov_high=0.040, w_alpha=0.3,
                 z_close=0.02, diverge_n=5, cov_threshold=0.05,
                 q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0):
        self.kp_xy = kp_xy               # 수평 P 게인
        self.kp_yaw = kp_yaw             # yaw P 게인 (현재 yaw=0 고정이라 미사용)
        self.v_max = v_max               # 수평 속도 상한
        self.descend_speed = descend_speed  # 하강 속도
        self.eps_descend = eps_descend   # 이 오차 이내여야 하강 시작
        self.eps_grasp = eps_grasp       # 폐합 판정 오차 임계
        self.n_stable = n_stable         # 폐합 판정에 필요한 연속 안정 주기 수
        # dt_latency에 2.3절의 "루프 반주기" 보정도 함께 흡수한다(1차 구현 단순화).
        self.dt_latency = dt_latency     # 지연 보상(lookahead) 시간
        self.timeout_s = timeout_s       # 서보 전체 타임아웃
        self.t_lost_s = t_lost_s         # 추적 유실 판정 시간
        self.innov_low = innov_low       # 이 잔차 이하 -> w=1(피드포워드 완전 신뢰)
        self.innov_high = innov_high     # 이 잔차 이상 -> w=0 + 속도 공분산 리셋
        self.w_alpha = w_alpha           # w의 저역통과 스무딩 계수
        self.z_close = z_close           # 폐합 판정 z_gap 임계
        self.diverge_n = diverge_n       # 발산 판정에 볼 연속 오차 개수
        self.cov_threshold = cov_threshold  # 폐합 판정용 속도 공분산 임계
        # 내부 KalmanXYZV로 그대로 전달되는 필터 노이즈 파라미터 (kalman.py 참고)
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_xy = r_xy
        self.r_z = r_z
        self.p0_vel_reset = p0_vel_reset

        self._state = ServoState.TRACKING
        # 실제 제어에 쓰이는 정밀 칼만 필터 (vision_node의 알파-베타 필터와 별개)
        self._filter = KalmanXYZV(
            q_pos=self.q_pos, q_vel=self.q_vel, r_xy=self.r_xy, r_z=self.r_z,
            p0_vel_reset=self.p0_vel_reset)
        self._w = 0.0                    # 피드포워드 신뢰 가중치(2.3절)
        self._last_track_time = None     # 마지막 ToolTrack의 header stamp (필터 dt 계산용)
        self._last_msg_time = None       # 마지막 ToolTrack을 "받은" 시각(monotonic) - t_lost 판정용
        self._start_time = None          # servo_pick 시작 시각(monotonic) - timeout 판정용
        self._stable_count = 0           # eps_grasp 이내로 연속 몇 주기째인지
        self._error_history = []         # 최근 e_xy 기록(발산 판정용)
        self._last_z_gap = None          # 마지막으로 계산한 |tcp_z - 목표 z|

    def start(self, tool_class, grasp_width_mm, grasp_force_n):
        """servo_pick goal 수신 시 1회 호출 - 모든 내부 상태를 초기화한다."""
        self.tool_class = tool_class
        self.grasp_width_mm = grasp_width_mm
        self.grasp_force_n = grasp_force_n
        self._state = ServoState.TRACKING
        self._filter = KalmanXYZV(
            q_pos=self.q_pos, q_vel=self.q_vel, r_xy=self.r_xy, r_z=self.r_z,
            p0_vel_reset=self.p0_vel_reset)
        self._w = 0.0
        self._last_track_time = None
        self._last_msg_time = None
        self._start_time = time.monotonic()
        self._stable_count = 0
        self._error_history = []
        self._last_z_gap = None

    def on_tool_track(self, msg):
        """/vision/tool_track 수신마다 호출 - 칼만 필터를 갱신하고 innovation으로
        피드포워드 가중치 w를 재계산한다(2.3/2.5절)."""
        now = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.position

        if not self._filter._initialized:
            # 이번 서보 사이클의 첫 관측 - 필터 초기화만 하고 넘어간다(예측할 이전 상태가 없음)
            self._filter.initialize(pos.x, pos.y, pos.z)
            self._last_track_time = stamp
            self._last_msg_time = now
            return

        dt = max(stamp - self._last_track_time, 1e-3)
        self._filter.predict(dt)

        # depth_valid 여부에 따라 z까지 갱신할지, x·y만 갱신할지 결정 (2.7절)
        if msg.depth_valid:
            innov_xy = self._filter.update_xyz([pos.x, pos.y, pos.z])
        else:
            innov_xy = self._filter.update_xy_only([pos.x, pos.y])

        # innovation(예측-관측 잔차) 크기로 w를 결정: 작으면 등속 가정 유효(w->1),
        # 크면 위반(w->0) + 속도 공분산 리셋으로 필터를 빠르게 재수렴시킴
        if innov_xy >= self.innov_high:
            self._filter.reset_velocity_covariance()
            w_target = 0.0
        elif innov_xy <= self.innov_low:
            w_target = 1.0
        else:
            span = self.innov_high - self.innov_low
            w_target = 1.0 - (innov_xy - self.innov_low) / span

        # w_target을 바로 쓰지 않고 저역통과 필터로 스무딩 - 잔차 노이즈로 인한 채터링 방지
        self._w = self.w_alpha * w_target + (1.0 - self.w_alpha) * self._w
        self._last_track_time = stamp
        self._last_msg_time = now

    def step(self, tcp_pose, now):
        """RT 명령 주기마다 호출 - PBVS 제어식(2.3절)으로 다음 속도 명령을 계산한다.
        tcp_pose: 현재 TCP pose(base_link 기준 x,y,z,rx,ry,rz). now: time.monotonic() 값."""
        if not self._filter._initialized:
            # 아직 ToolTrack을 한 번도 못 받았으면 정지 명령
            return ServoCommand()

        # p_ref = 필터 추정 위치를 Δt_lat만큼 앞으로 외삽한 "지금 목표로 삼아야 할 위치"
        p_ref = self._filter.predict_position(self.dt_latency)
        v_tool = self._filter.velocity  # 피드포워드 항(v̂_tool)

        tcp_x, tcp_y, tcp_z = tcp_pose[0], tcp_pose[1], tcp_pose[2]
        e_x = p_ref[0] - tcp_x
        e_y = p_ref[1] - tcp_y
        e_xy_norm = float(np.hypot(e_x, e_y))

        # 발산 판정용 오차 이력 (최근 diverge_n개만 유지)
        self._error_history.append(e_xy_norm)
        if len(self._error_history) > self.diverge_n:
            self._error_history.pop(0)

        self._last_z_gap = abs(tcp_z - p_ref[2])

        # 핵심 제어식: v_cmd = w·v̂_tool + Kp·e (속도 피드포워드 + P 피드백)
        vx = self._w * v_tool[0] + self.kp_xy * e_x
        vy = self._w * v_tool[1] + self.kp_xy * e_y
        speed = float(np.hypot(vx, vy))
        if speed > self.v_max:
            scale = self.v_max / speed
            vx *= scale
            vy *= scale

        # 하강은 별도 프로파일: 수평 오차가 충분히 작을 때만 진행, 아니면 대기(수평 정렬 우선)
        if e_xy_norm < self.eps_descend:
            self._state = ServoState.DESCENDING
            vz = -self.descend_speed
        else:
            self._state = ServoState.TRACKING
            vz = 0.0

        # 폐합 판정용 안정 카운터 - eps_grasp 이내가 아니면 리셋(연속성 요구)
        if e_xy_norm < self.eps_grasp:
            self._stable_count += 1
        else:
            self._stable_count = 0

        return ServoCommand(vx=vx, vy=vy, vz=vz, yaw_rate=0.0)

    def get_state(self):
        """RobotTask Feedback.state로 그대로 노출되는 현재 서보 상태."""
        return self._state

    def should_close(self):
        """그리퍼 폐합 판정(2.6절) - 세 조건 모두 만족해야 True:
        오차가 n_stable주기 연속 충분히 작고, z까지 충분히 가깝고, 필터가 수렴했을 것."""
        if self._stable_count < self.n_stable:
            return False
        if self._last_z_gap is None or self._last_z_gap >= self.z_close:
            return False
        if self._filter.velocity_covariance_trace >= self.cov_threshold:
            return False
        self._state = ServoState.CLOSING
        return True

    def should_abort(self):
        """중단 판정(2.8절) - 사유 문자열 또는 정상이면 None.
        robot_control_node는 이 문자열을 RobotTask.Result.message에 그대로 담아 반환하고,
        task_manager는 그 문자열에 'torque'가 있는지로 FAULT/재시도를 나눈다."""
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
