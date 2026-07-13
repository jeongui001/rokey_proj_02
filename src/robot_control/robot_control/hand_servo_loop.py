import time

import numpy as np

from robot_control.servo_loop import ServoCommand


class HandServoState:
    APPROACHING = 'approaching'
    FOLLOWING = 'following'
    STOPPING = 'stopping'


class HandServoLoop:
    """handover_approach RT 서보 루프 - servo_loop.ServoLoop과 대칭 구조(전체 계획.md
    2절 calculate 모듈)이지만, 손은 공구처럼 등속 파지 이동을 예측할 필요가 없어
    ServoLoop의 KalmanXYZV 없이 최근 HandTrack 관측만으로 목표를 계산한다.

    팀원1(robot_control_node)이 이 클래스를 부르는 순서(_execute_handover_approach와 동일):
      1. handover_approach goal 수신 시 start() 한 번
      2. /vision/hand_track 콜백마다 on_hand_track(msg)
      3. RT 명령 주기마다 step(tcp_pose, now) - 제어 명령 계산
      4. 매 틱마다 tick()으로 종료 조건 확인
    """

    def __init__(self, kp_xy, kp_z, v_max, offset_m, t_lost_s, timeout_s):
        self.kp_xy = kp_xy         # 수평(x,y) P 게인
        self.kp_z = kp_z           # 수직(z) P 게인
        self.v_max = v_max         # 각 축 속도 상한(사람 접근이라 보수적으로 잡는다)
        self.offset_m = offset_m   # TCP->손 방향 위, 손 앞 이 거리(m)만큼 못 미친 지점을 목표로 한다
        self.t_lost_s = t_lost_s   # 손 유실 판정 시간
        self.timeout_s = timeout_s  # 서보 전체 타임아웃(0이면 비활성 - 주먹까지 계속 추종)

        self._state = HandServoState.APPROACHING
        self._hand_pos = None        # 최근 손 위치(base_link, m) - [x, y, z]
        self._detected = False       # 최근 HandTrack.detected
        self._fist = False           # 최근 HandTrack.fist(주먹 확정)
        self._last_msg_time = None   # 마지막 HandTrack을 "받은" 시각(monotonic) - t_lost 판정용
        self._start_time = None      # handover_approach 시작 시각(monotonic) - timeout 판정용
        self._last_command = ServoCommand()  # DEBUG_LOG: 최근 속도 명령
        self._last_target = None     # DEBUG_LOG: 최근 목표점(m)

    def start(self):
        """handover_approach goal 수신 시 1회 호출 - 모든 내부 상태를 초기화한다."""
        self._state = HandServoState.APPROACHING
        self._hand_pos = None
        self._detected = False
        self._fist = False
        self._last_msg_time = None
        self._start_time = time.monotonic()
        self._last_command = ServoCommand()
        self._last_target = None

    def on_hand_track(self, msg):
        """/vision/hand_track 수신마다 호출 - 최근 손 위치·검출·주먹 상태를 저장한다."""
        pos = msg.pose.position
        self._hand_pos = np.array([pos.x, pos.y, pos.z], dtype=float)
        self._detected = bool(msg.detected)
        self._fist = bool(msg.fist)
        self._last_msg_time = time.monotonic()

    def step(self, tcp_pose, now):
        """RT 명령 주기마다 호출 - TCP->손 방향 위 offset_m 지점을 목표로 P 제어한다.
        tcp_pose: 현재 TCP pose(base_link 기준 x,y,z,rx,ry,rz, m). now: time.monotonic() 값."""
        if self._hand_pos is None or not self._detected:
            # 아직 손을 한 번도 못 봤거나 이번 프레임에 미검출 - 정지 명령
            self._last_command = ServoCommand()
            return self._last_command

        tcp = np.array(tcp_pose[:3], dtype=float)
        direction = self._hand_pos - tcp
        distance = float(np.linalg.norm(direction))
        unit = direction / distance if distance > 1e-6 else np.zeros(3)
        target = self._hand_pos - self.offset_m * unit
        self._last_target = target

        e = target - tcp
        vx = self.kp_xy * e[0]
        vy = self.kp_xy * e[1]
        vz = self.kp_z * e[2]
        # 축별로 따로 클리핑하면 한 축만 먼저 포화될 때 명령 벡터 방향이 실제
        # 목표 방향에서 벗어나 로봇이 손으로 곧장 오지 않고 휘어서 접근한다 -
        # servo_pick(ServoLoop.step)처럼 벡터 크기 하나로 묶어 클리핑해 방향을
        # 보존한다.
        speed = float(np.linalg.norm([vx, vy, vz]))
        if speed > self.v_max:
            scale = self.v_max / speed
            vx *= scale
            vy *= scale
            vz *= scale

        self._state = (
            HandServoState.STOPPING if self._fist else HandServoState.FOLLOWING)
        self._last_command = ServoCommand(vx=vx, vy=vy, vz=vz, yaw_rate=0.0)
        return self._last_command

    def tick(self):
        """종료 판정 - ('STOP'|'ABORT'|'CONTINUE', reason). robot_control_node의
        _run_rt_tracking은 'STOP'을 servo_pick의 'CLOSE'와 동일하게(=ARRIVED) 취급한다."""
        if self._fist:
            self._state = HandServoState.STOPPING
            return ('STOP', 'fist_detected')
        if (self._last_msg_time is not None
                and time.monotonic() - self._last_msg_time > self.t_lost_s):
            return ('ABORT', 'hand_lost')
        if (self.timeout_s > 0
                and self._start_time is not None
                and time.monotonic() - self._start_time > self.timeout_s):
            return ('ABORT', 'timeout')
        return ('CONTINUE', None)

    def get_state(self):
        """RobotTask Feedback.state로 그대로 노출되는 현재 서보 상태."""
        return self._state

    def debug_snapshot(self):
        """DEBUG_LOG: ROS 노드가 고주기 로그를 throttle해서 남길 때 쓰는 서보 내부 수치."""
        return {
            'state': self._state,
            'detected': self._detected,
            'fist': self._fist,
            'hand_pos_m': None if self._hand_pos is None else [float(v) for v in self._hand_pos],
            'target_m': None if self._last_target is None else [float(v) for v in self._last_target],
            'cmd_m_s': {
                'vx': float(self._last_command.vx),
                'vy': float(self._last_command.vy),
                'vz': float(self._last_command.vz),
                'yaw_rate': float(self._last_command.yaw_rate),
            },
        }
