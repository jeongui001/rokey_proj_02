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


def _wrap_yaw_error_deg(target_deg, current_deg):
    """대칭 2-핑거 그리퍼(180도 주기) 기준 최단 경로 각도 오차(deg, [-90,90])를 구한다."""
    return ((target_deg - current_deg + 90.0) % 180.0) - 90.0


def _zyz_deg_to_rot(a_deg, b_deg, c_deg):
    """Doosan posx의 ZYZ 오일러 각(deg) -> 3x3 회전행렬. R = Rz(A) @ Ry(B) @ Rz(C).

    grasp_geometry.zyz_deg_to_rot와 동일 컨벤션이지만 여기 로컬로 복제한다 -
    robot_control이 vision_node 패키지를 import하지 않는 기존 경계를 유지하기
    위함(_yaw_from_quaternion처럼 이 파일이 이미 쓰는 "작은 순수-수학 헬퍼는
    패키지 경계를 넘기지 않고 로컬 복제" 패턴)."""
    a, b, c = np.deg2rad([a_deg, b_deg, c_deg])
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rz_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry_b = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz_c = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz_a @ ry_b @ rz_c


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
                 q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0,
                 n_stable_z=5, diverge_min_delta_m=0.01, descend_accel_m_s2=0.1,
                 descend_stop_margin_m=0.0, v_grasp_max=0.05, n_stable_v=5,
                 v_tool_deadband_m_s=0.03, yaw_rate_max_deg_s=30.0,
                 eps_yaw_deg=5.0, n_stable_yaw=5, yaw_sign=1.0, yaw_offset_deg=0.0,
                 diverge_n_yaw=15, diverge_min_delta_deg=10.0):
        self.kp_xy = kp_xy               # 수평 P 게인
        self.kp_yaw = kp_yaw             # yaw P 게인 - deg 오차 -> deg/s 명령
        self.v_max = v_max               # 수평 속도 상한
        self.yaw_rate_max_deg_s = yaw_rate_max_deg_s  # 회전 속도 상한(v_max와 별개 단위)
        self.eps_yaw_deg = eps_yaw_deg   # 폐합 판정 각도 오차 임계(deg)
        self.n_stable_yaw = n_stable_yaw  # 폐합 판정에 필요한 연속 안정 주기 수(z_close와 동일 방식)
        # 카메라 base-frame grip_deg(0=base_link X축 기준) <-> TCP posx C각(ZYZ, deg)
        # 사이의 부호/오프셋 - 실기 캘리브레이션 전까지는 항등(1.0/0.0)이며 이 관계가
        # 실제로 맞는지는 hardware_ready 게이트를 열기 전 반드시 검증해야 한다.
        self.yaw_sign = yaw_sign
        self.yaw_offset_deg = yaw_offset_deg
        self.diverge_n_yaw = diverge_n_yaw  # yaw 발산 판정에 볼 연속 오차 개수(xy의 diverge_n과 대칭)
        # diverge_n_yaw틱 연속 증가에 더해 요구하는 최소 총 증가폭(deg) - xy의
        # diverge_min_delta_m과 같은 이유(노이즈성 연속 증가 걸러내기).
        self.diverge_min_delta_deg = diverge_min_delta_deg
        self.descend_speed = descend_speed  # 하강 속도
        # speedl에 실제로 걸리는 가속도 제한(servo_pick.speedl_acc_trans_mm_s2와 동일
        # 값, m/s² 단위) - vz=0을 명령해도 로봇이 이 가속도로만 감속하므로(2026-07-10
        # 실기: descend_speed=0.1m/s에서 커맨드 0 이후 약 50mm 관성 하강 후 바닥 충돌
        # 확인, v²/(2a)=0.1²/(2*0.1)=0.05m로 물리 계산과 일치), z_gap 기준으로 미리
        # 감속해야 z_close 문턱과 무관하게 실제 정지 지점이 목표에 가까워진다.
        self.descend_accel_m_s2 = descend_accel_m_s2
        # safe_speed 공식(v=sqrt(2*a*거리))이 목표(z_gap=0, 즉 물리 접촉)에서 속도 0을
        # 겨냥하면, z_close 문턱 자체가 "락 이후 예상 관성 오버슈트"로 고스란히
        # 소진된다(2026-07-11 실기: z_close=0.03에서 락 시점 속도 ~77mm/s, 관성거리
        # ~30mm=z_close와 거의 일치, 정지 물체인데도 바닥 충돌 확인 - 목표 z가 이미
        # 테이블면에 가까운 얇은 도구라 여유가 전혀 없었음). 이 마진(m)만큼 목표를
        # 미리 띄워 제동 곡선이 z_gap=margin에서 속도 0을 겨냥하게 해, 락 이후에도
        # 표면 위 margin만큼 여유가 남도록 한다. z_close보다 작아야 한다(같거나 크면
        # z_gap이 z_close 밑으로 내려가기 전에 이미 vz=0에 수렴해 폐합 조건이 영원히
        # 안 걸릴 수 있다).
        self.descend_stop_margin_m = descend_stop_margin_m
        self.eps_descend = eps_descend   # 이 오차 이내여야 하강 시작
        self.eps_grasp = eps_grasp       # 폐합 판정 오차 임계
        self.n_stable = n_stable         # 폐합 판정에 필요한 연속 안정 주기 수
        self.n_stable_z = n_stable_z     # z_close 판정에 필요한 연속 안정 주기 수(단발 depth 노이즈 방지)
        # dt_latency에 2.3절의 "루프 반주기" 보정도 함께 흡수한다(1차 구현 단순화).
        self.dt_latency = dt_latency     # 지연 보상(lookahead) 시간
        self.timeout_s = timeout_s       # 서보 전체 타임아웃
        self.t_lost_s = t_lost_s         # 추적 유실 판정 시간
        self.innov_low = innov_low       # 이 잔차 이하 -> w=1(피드포워드 완전 신뢰)
        self.innov_high = innov_high     # 이 잔차 이상 -> w=0 + 속도 공분산 리셋
        self.w_alpha = w_alpha           # w의 저역통과 스무딩 계수
        self.z_close = z_close           # 폐합 판정 z_gap 임계
        self.diverge_n = diverge_n       # 발산 판정에 볼 연속 오차 개수
        # diverge_n틱 연속 증가만으로는 부족하고, 그 구간 총 증가폭도 이 값(m) 이상이어야
        # 발산으로 본다 - RT 루프(100Hz)가 비전(~55~60Hz)보다 빨라 비전 갱신 사이에는
        # lead_time(dt_latency+elapsed_since_track) 외삽만으로도 오차가 매 틱 미세하게
        # 늘어나는 구간이 정상적으로 생긴다(2026-07-10 실기: 정지 물체에서도 diverging
        # 오탐 확인). 노이즈 수준의 연속 증가를 걸러내되 진짜 발산은 여전히 잡기 위한
        # 최소 증가폭 문턱.
        self.diverge_min_delta_m = diverge_min_delta_m
        self.cov_threshold = cov_threshold  # 폐합 판정용 속도 공분산 임계
        self.v_grasp_max = v_grasp_max   # 폐합 판정용 공구 속도(xy, m/s) 상한
        self.n_stable_v = n_stable_v     # v_grasp_max 판정에 필요한 연속 안정 주기 수(z_close와 동일 방식)
        # 정지 물체도 카메라 노이즈만으로 v_tool이 완전히 0은 아니게 추정된다(실측
        # 노이즈 바닥: 정지 렌치 재생 시 p95=10.6mm/s, max=23.2mm/s). r_xy를 실측치로
        # 낮춘 뒤 이 노이즈가 w(피드포워드 신뢰도)가 1 근처로 빨리 수렴하는 정지
        # 상황에서 그대로 vx/vy 명령에 실려 로봇이 떨리는 현상 확인(2026-07-12 실기) -
        # tool_speed가 이 값 미만이면 피드포워드 기여를 선형으로 줄여 순수 P제어에
        # 가깝게 만든다. 실제 이동(수십~수백mm/s)에는 영향 없도록 노이즈 바닥보다
        # 확실히 크게 잡았다.
        self.v_tool_deadband_m_s = v_tool_deadband_m_s
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
        self._z_stable_count = 0         # z_close 이내로 연속 몇 주기째인지
        self._v_stable_count = 0         # v_grasp_max 이내로 연속 몇 주기째인지
        self._error_history = []         # 최근 e_xy 기록(발산 판정용)
        self._last_z_gap = None          # 마지막으로 계산한 |tcp_z - 목표 z|
        self._grasp_locked = False       # should_close() 만족 이후 z를 영구 고정할지
        self._z_locked = False           # z_stable_count가 n_stable_z 도달 이후 z를 영구 고정할지
        self._last_e_xy_norm = None      # DEBUG_LOG: 최근 xy 오차(m)
        self._last_command = ServoCommand()  # DEBUG_LOG: 최근 속도 명령
        self._last_innovation_xy = None  # DEBUG_LOG: 최근 Kalman innovation(m)
        self._last_w_target = None       # DEBUG_LOG: 최근 feed-forward 목표 가중치
        self._last_depth_valid = None    # DEBUG_LOG: 최근 ToolTrack depth_valid
        self._last_tool_speed = None     # DEBUG_LOG: 최근 공구 xy 속력(m/s)
        self._yaw_target_deg = None      # 최근 유효 grip yaw 목표(deg, mod 180) - yaw_valid=False 프레임은 hold
        self._yaw_stable_count = 0       # eps_yaw_deg 이내로 연속 몇 주기째인지
        self._last_yaw_error_deg = None  # DEBUG_LOG: 최근 yaw 오차(deg)
        self._yaw_error_history = []     # 최근 |yaw 오차| 기록(발산 판정용)

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
        self._z_stable_count = 0
        self._v_stable_count = 0
        self._error_history = []
        self._last_z_gap = None
        self._grasp_locked = False
        self._z_locked = False
        self._last_e_xy_norm = None
        self._last_command = ServoCommand()
        self._last_innovation_xy = None
        self._last_w_target = None
        self._last_depth_valid = None
        self._last_tool_speed = None
        self._yaw_target_deg = None
        self._yaw_stable_count = 0
        self._last_yaw_error_deg = None
        self._yaw_error_history = []

    def on_tool_track(self, msg):
        """/vision/tool_track 수신마다 호출 - 칼만 필터를 갱신하고 innovation으로
        피드포워드 가중치 w를 재계산한다(2.3/2.5절)."""
        now = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.position

        # yaw는 위치 필터 초기화 여부와 무관하게 탐지된 첫 프레임부터 즉시 목표를
        # 갱신한다 - "완전히 내려간 뒤 6축을 돌리는" 대신 서보잉과 동시에 각도를
        # 맞춰가기 위함(yaw_valid=False인 프레임은 직전 목표를 그대로 유지=hold).
        if bool(getattr(msg, 'yaw_valid', False)):
            self._yaw_target_deg = math.degrees(_yaw_from_quaternion(msg.pose.orientation)) % 180.0

        if not self._filter._initialized:
            # 이번 서보 사이클의 첫 관측 - 필터 초기화만 하고 넘어간다(예측할 이전 상태가 없음)
            self._filter.initialize(pos.x, pos.y, pos.z)
            self._last_track_time = stamp
            self._last_msg_time = now
            self._last_depth_valid = bool(msg.depth_valid)
            return

        dt = max(stamp - self._last_track_time, 1e-3)
        self._filter.predict(dt)

        # depth_valid 여부에 따라 z까지 갱신할지, x·y만 갱신할지 결정 (2.7절)
        if msg.depth_valid:
            innov_xy = self._filter.update_xyz([pos.x, pos.y, pos.z])
        else:
            innov_xy = self._filter.update_xy_only([pos.x, pos.y])
        self._last_innovation_xy = float(innov_xy)
        self._last_depth_valid = bool(msg.depth_valid)

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
        self._last_w_target = float(w_target)

        # w_target을 바로 쓰지 않고 저역통과 필터로 스무딩 - 잔차 노이즈로 인한 채터링 방지
        self._w = self.w_alpha * w_target + (1.0 - self.w_alpha) * self._w
        self._last_track_time = stamp
        self._last_msg_time = now

    def step(self, tcp_pose, now):
        """RT 명령 주기마다 호출 - PBVS 제어식(2.3절)으로 다음 속도 명령을 계산한다.
        tcp_pose: 현재 TCP pose(base_link 기준 x,y,z(m),rx,ry,rz(deg) - posx 6-vector 그대로).
        now: time.monotonic() 값."""
        if not self._filter._initialized:
            # 아직 ToolTrack을 한 번도 못 받았으면 정지 명령
            self._last_command = ServoCommand()
            return self._last_command

        # yaw 제어는 ServoState(TRACKING/DESCENDING)와 무관하게 매 틱 독립적으로
        # 돈다 - "완전히 내려간 뒤 6축을 돌리는" 대신 xy/z 서보잉과 같은 speedl
        # 틱에 실어 보낸다(추가 루프/통신 왕복 없음). tcp_pose[3:6] = posx (A,B,C)
        # ZYZ 오일러(deg).
        #
        # "현재 그립 각도"를 C 성분 그대로 읽지 않는다 - 이 파이프라인의 top-down
        # 파지 자세는 named_poses.watch 주석의 base 좌표 참고값(B≈178.53°)에서 보듯
        # ZYZ 오일러의 짐벌락(B=180°) 특이점 바로 근처에서 동작하는데, 그 근방에서는
        # 손목의 미세한 실제 회전이 C 성분에는 거대한 값 요동으로 나타날 수 있다
        # (오일러각 분해 자체의 수치적 불안정성). 2026-07-13 실기에서 이 raw C 값을
        # 그대로 P제어 피드백에 썼을 때 필요한 것보다 훨씬 큰 회전(과회전)이 확인돼,
        # 회전행렬을 완전히 조립한 뒤 기준 벡터를 그 행렬로 회전시켜 base 평면에
        # 투영하는 방식으로 바꿨다 - vision측 _grip_deg_to_base_quaternion과 동일한
        # 기법이라 짐벌락 근방에서도 연속적이다. B=0일 때는 수학적으로 기존 raw-C
        # 방식과 완전히 동치(Ry(0)=단위행렬)이므로 named pose가 B=0에 가까운
        # 환경에서는 동작이 그대로 유지된다.
        if self._yaw_target_deg is not None and len(tcp_pose) >= 6:
            rot = _zyz_deg_to_rot(tcp_pose[3], tcp_pose[4], tcp_pose[5])
            offset_rad = np.deg2rad(self.yaw_offset_deg)
            ref = np.array([np.cos(offset_rad), np.sin(offset_rad), 0.0])
            axis_base = rot @ ref
            current_grip_deg = (self.yaw_sign
                                 * np.degrees(np.arctan2(axis_base[1], axis_base[0]))) % 180.0
            yaw_error_deg = _wrap_yaw_error_deg(self._yaw_target_deg, current_grip_deg)
            self._last_yaw_error_deg = yaw_error_deg
            yaw_rate = _clip(self.kp_yaw * yaw_error_deg, self.yaw_rate_max_deg_s)
            if abs(yaw_error_deg) < self.eps_yaw_deg:
                self._yaw_stable_count += 1
            else:
                self._yaw_stable_count = 0
            # yaw 발산 판정용 오차 이력(절댓값) - xy의 _error_history와 대칭 패턴.
            # 부호 있는 값이 아니라 절댓값을 쌓는다 - yaw 오차는 wrap 경계를 넘으며
            # 부호가 바뀔 수 있어 xy처럼 부호 있는 값의 단조 증가를 보면 오탐/누락이
            # 생긴다.
            self._yaw_error_history.append(abs(yaw_error_deg))
            if len(self._yaw_error_history) > self.diverge_n_yaw:
                self._yaw_error_history.pop(0)
        else:
            yaw_rate = 0.0
            self._last_yaw_error_deg = None
            self._yaw_stable_count = 0

        # p_ref = 필터 추정 위치를 (Δt_lat + 마지막 ToolTrack 이후 실제 경과시간)만큼
        # 앞으로 외삽한 "지금 목표로 삼아야 할 위치". 비전 갱신이 뜸해질수록 더 멀리
        # 내다봐야 목표점이 얼어붙지 않는다.
        elapsed_since_track = max(now - self._last_msg_time, 0.0)
        lead_time = self.dt_latency + elapsed_since_track
        p_ref = self._filter.predict_position(lead_time)
        v_tool = self._filter.velocity  # 피드포워드 항(v̂_tool)
        tool_speed = float(np.hypot(v_tool[0], v_tool[1]))
        self._last_tool_speed = tool_speed
        # v_tool_deadband_m_s 미만이면 피드포워드 기여를 선형으로 줄인다(0~1 램프) -
        # 정지 물체의 노이즈성 v_tool이 그대로 명령에 실려 떨리는 것을 막기 위함.
        ff_scale = min(tool_speed / self.v_tool_deadband_m_s, 1.0) \
            if self.v_tool_deadband_m_s > 0 else 1.0

        tcp_x, tcp_y, tcp_z = tcp_pose[0], tcp_pose[1], tcp_pose[2]
        e_x = p_ref[0] - tcp_x
        e_y = p_ref[1] - tcp_y
        e_xy_norm = float(np.hypot(e_x, e_y))
        self._last_e_xy_norm = e_xy_norm

        # 발산 판정용 오차 이력 (최근 diverge_n개만 유지)
        self._error_history.append(e_xy_norm)
        if len(self._error_history) > self.diverge_n:
            self._error_history.pop(0)

        self._last_z_gap = abs(tcp_z - p_ref[2])

        # 핵심 제어식: v_cmd = w·ff_scale·v̂_tool + Kp·e (속도 피드포워드 + P 피드백)
        vx = self._w * ff_scale * v_tool[0] + self.kp_xy * e_x
        vy = self._w * ff_scale * v_tool[1] + self.kp_xy * e_y
        speed = float(np.hypot(vx, vy))
        if speed > self.v_max:
            scale = self.v_max / speed
            vx *= scale
            vy *= scale

        # z_close 이내 판정도 xy의 stable_count와 같은 방식으로 디바운스한다 - depth
        # 관측 한 프레임이 노이즈로 튀어 z_gap이 순간적으로 z_close 밑으로 떨어져도
        # 그 한 번만으로 하강을 멈추지 않도록, n_stable_z주기 연속으로 z_close
        # 이내여야 "도착"으로 인정한다.
        if self._last_z_gap < self.z_close:
            self._z_stable_count += 1
        else:
            self._z_stable_count = 0

        # 공구 속도(xy 평면) 판정 - z_close와 같은 방식으로 디바운스한다: 필터
        # 속도 추정치는 단발 프레임에서 튈 수 있으므로, n_stable_v주기 연속으로
        # v_grasp_max 이내여야 "충분히 느림"으로 인정한다. (tool_speed는 위에서
        # ff_scale 계산 시 이미 구했다.)
        if tool_speed < self.v_grasp_max:
            self._v_stable_count += 1
        else:
            self._v_stable_count = 0

        # 하강은 별도 프로파일: 수평 오차가 충분히 작을 때만 진행, 아니면 대기(수평 정렬 우선).
        # z가 안정적으로 z_close 이내에 들어오면 xy 안정성/공분산(should_close 조건)과
        # 무관하게 즉시 vz를 0으로 고정한다 - descend_speed는 비례 제어가 아니라 상수
        # 속도라서 should_close()의 복합 조건(폐합 가능 여부)이 늦게 만족돼도 목표 z를
        # 지나쳐 계속 하강하면 안 되기 때문이다.
        # _z_stable_count가 n_stable_z에 한 번 도달하면(_z_locked) 이후 z_gap이 노이즈로
        # 다시 벌어지더라도 재하강하지 않도록 영구히 고정한다. should_close()가 한 번이라도
        # True가 된 뒤(_grasp_locked)에는 그리퍼가 실제로 닫히는 동안에도 xy 추적은 계속하되,
        # z는 마찬가지로 절대 재하강하지 않도록 영구히 고정한다 - 그렇지 않으면 그리퍼 폐합
        # 도중 depth 노이즈 등으로 z_gap이 잠시 벌어졌을 때 DESCENDING으로 되돌아가 이미
        # 폐합 판정을 마친 물체를 다시 밀고 내려가게 된다.
        if self._z_stable_count >= self.n_stable_z:
            self._z_locked = True
        if self._grasp_locked:
            vz = 0.0
        elif self._z_locked:
            self._state = ServoState.TRACKING
            vz = 0.0
        elif e_xy_norm < self.eps_descend:
            self._state = ServoState.DESCENDING
            # 남은 z_gap 안에서 descend_accel_m_s2로 정지 가능한 속도로 상한을 건다
            # (v = sqrt(2*a*거리)) - 그래야 vz=0 명령이 나가는 시점(z_locked)에 이미
            # 속도가 충분히 낮아, 가속도 제한 때문에 생기는 관성 하강 거리가 z_close
            # 수준으로 줄어든다. 그 전(z_gap이 클 때)에는 descend_speed로 상한이 걸린다.
            brake_distance = max(self._last_z_gap - self.descend_stop_margin_m, 0.0)
            safe_speed = math.sqrt(2.0 * self.descend_accel_m_s2 * brake_distance)
            vz = -min(self.descend_speed, safe_speed)
        else:
            self._state = ServoState.TRACKING
            vz = 0.0

        # 폐합 판정용 안정 카운터 - eps_grasp 이내가 아니면 리셋(연속성 요구)
        if e_xy_norm < self.eps_grasp:
            self._stable_count += 1
        else:
            self._stable_count = 0

        self._last_command = ServoCommand(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
        return self._last_command

    def get_state(self):
        """RobotTask Feedback.state로 그대로 노출되는 현재 서보 상태."""
        return self._state

    def should_close(self):
        """그리퍼 폐합 판정(2.6절) - 다음 조건 모두 만족해야 True:
        오차가 n_stable주기 연속 충분히 작고, z까지 충분히 가깝고, 필터가 수렴했고,
        공구 속도가 n_stable_v주기 연속 충분히 느리고, yaw 목표가 있다면 그마저도
        n_stable_yaw주기 연속 충분히 정렬됐을 것(목표가 한 번도 없었으면 이 조건은
        건너뛴다 - 비전 yaw가 아예 실패해도 파지 자체는 막지 않기 위함)."""
        if self._stable_count < self.n_stable:
            return False
        if self._last_z_gap is None or self._z_stable_count < self.n_stable_z:
            return False
        if self._filter.velocity_covariance_trace >= self.cov_threshold:
            return False
        if self._yaw_target_deg is not None and self._yaw_stable_count < self.n_stable_yaw:
            return False
        if self._v_stable_count < self.n_stable_v:
            return False
        self._state = ServoState.CLOSING
        self._grasp_locked = True
        return True

    def should_abort(self):
        """중단 판정(2.8절) - 사유 문자열 또는 정상이면 None.
        robot_control_node는 이 문자열을 RobotTask.Result.message에 그대로 담아 반환하고,
        task_manager는 그 문자열에 'torque'가 있는지로 FAULT/재시도를 나눈다."""
        if self._start_time is not None and time.monotonic() - self._start_time > self.timeout_s:
            return 'timeout'
        if self._last_msg_time is not None and time.monotonic() - self._last_msg_time > self.t_lost_s:
            return 'tracking_lost'
        if (len(self._error_history) == self.diverge_n and all(
                self._error_history[i] < self._error_history[i + 1]
                for i in range(len(self._error_history) - 1))
                and self._error_history[-1] - self._error_history[0]
                >= self.diverge_min_delta_m):
            return 'diverging'
        # yaw 발산 판정 - xy와 대칭 패턴이지만 절댓값 기준 단조 증가를 본다(_yaw_error_history는
        # step()에서 이미 절댓값으로 쌓임 - wrap 경계를 넘으며 부호가 바뀌는 것과 무관하게
        # 판정하기 위함). raw C 오일러 성분을 회전행렬 기반 계산으로 바꿔 짐벌락 근방
        # 불안정성은 줄였지만, 캘리브레이션 미확정(yaw_sign/yaw_offset_deg) 등 남은 오차
        # 원인으로도 회전이 계속 커질 수 있어 물리적 과회전 폭 자체를 여기서 한 번 더
        # 제한한다(2026-07-13 실기 과회전 확인 후 추가).
        if (len(self._yaw_error_history) == self.diverge_n_yaw and all(
                self._yaw_error_history[i] < self._yaw_error_history[i + 1]
                for i in range(len(self._yaw_error_history) - 1))
                and self._yaw_error_history[-1] - self._yaw_error_history[0]
                >= self.diverge_min_delta_deg):
            return 'yaw_diverging'
        # 참고: "공구가 방향 전환하여 시야 이탈 예상"(2.8절)은 판정 기준이 모호해
        # 과설계 우려가 있으므로 1차 구현 범위에서 제외했다. 실측 후 필요하면 추가한다.
        return None

    def debug_snapshot(self):
        """DEBUG_LOG: ROS 노드가 고주기 로그를 throttle해서 남길 때 쓰는 서보 내부 수치."""
        position = None
        velocity = None
        if self._filter._initialized:
            position = [float(v) for v in self._filter.position]
            velocity = [float(v) for v in self._filter.velocity]
        return {
            'state': self._state,
            'filter_initialized': bool(self._filter._initialized),
            'w': float(self._w),
            'w_target': self._last_w_target,
            'innovation_xy_m': self._last_innovation_xy,
            'e_xy_norm_m': self._last_e_xy_norm,
            'stable_count': self._stable_count,
            'last_z_gap_m': self._last_z_gap,
            'z_stable_count': self._z_stable_count,
            # descend_stop_margin_m이 z_close보다 크거나 같으면 z_gap이 z_close 밑으로
            # 내려가기도 전에 제동 곡선이 이미 vz=0을 겨냥해 z_stable_count가 영원히
            # 0에 머무는 교착이 생긴다(2026-07-13 실기 확인 - 그리퍼가 절대 안 닫힘).
            # 매 틱 계산이라기보다 파라미터 조합 자체의 정적 점검이지만, 로그에서
            # z_stable_count=0이 반복될 때 원인을 바로 알 수 있도록 여기 같이 남긴다.
            'z_close_margin_ok': self.descend_stop_margin_m < self.z_close,
            'tool_speed_m_s': self._last_tool_speed,
            'v_stable_count': self._v_stable_count,
            'velocity_covariance_trace': float(self._filter.velocity_covariance_trace),
            'error_history_m': [float(v) for v in self._error_history],
            'depth_valid': self._last_depth_valid,
            'position_m': position,
            'velocity_m_s': velocity,
            'yaw_target_deg': self._yaw_target_deg,
            'yaw_error_deg': self._last_yaw_error_deg,
            'yaw_stable_count': self._yaw_stable_count,
            'yaw_error_history_deg': [float(v) for v in self._yaw_error_history],
            'cmd_m_s': {
                'vx': float(self._last_command.vx),
                'vy': float(self._last_command.vy),
                'vz': float(self._last_command.vz),
                'yaw_rate': float(self._last_command.yaw_rate),
            },
        }
