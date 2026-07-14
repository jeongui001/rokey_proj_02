import json
import os
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import GripperState

from robot_control.doosan_driver import DoosanDriver, DoosanRobotControl
from robot_control.drfl_force_monitor import DrflForceMonitor
from robot_control.rg2_client import RG2Client, RG2Status
from robot_control.safety_monitor import (
    DoosanRobotState,
    FaultPrefix,
    SafetyMonitor,
    SafetyState,
)
from robot_control.hand_servo_loop import HandServoLoop
from robot_control.servo_loop import ServoLoop
from robot_control.task_executor import TaskExecutor

NAMED_POSE_NAMES = ('home', 'front', 'up', 'down', 'watch', 'handover_safe')

# servo_pick/release_and_retry, front/up/down 수동 이동은 체크리스트 항목이 아니라 제외
_GOAL_SENT_CHECKPOINTS = {
    ('move_named', 'watch'): ('B', 'move_watch_goal_sent'),
    ('move_named', 'handover_safe'): ('F', 'handover_safe_goal_sent'),
    ('move_named', 'home'): ('J', 'home_goal_sent'),
    ('handover_approach', ''): ('H', 'handover_approach_goal_sent'),
    ('handover_hold', ''): ('I', 'handover_hold_goal_sent'),
}


def _declare_double_array(node, name, default):
    if not default:
        # rclpy Humble 버그: 빈 리스트 기본값은 BYTE_ARRAY로 오추론되므로 Parameter.Type을
        # 직접 지정 - 대신 미초기화로 남으므로 읽는 쪽(_refresh_named_poses)에서 보정 필요.
        node.declare_parameter(name, Parameter.Type.DOUBLE_ARRAY)
        return
    node.declare_parameter(name, default)


class RobotControlNode(Node, TaskExecutor):
    def __init__(self):
        super().__init__('robot_control')

        self.declare_parameter('hardware_enabled', False)
        self.declare_parameter('robot_id', 'dsr01')
        self.declare_parameter('rg2_ip', '192.168.1.1')
        self.declare_parameter('rg2_port', 502)
        self.declare_parameter('rg2_gripper', 'rg2')
        self.declare_parameter('rg2.command_timeout_s', 5.0)
        self.declare_parameter('rg2.poll_interval_s', 0.05)
        self.declare_parameter('rg2.open_width_tolerance_mm', 2.0)
        self.declare_parameter('rg2.connect_timeout_s', 2.0)
        # COMMUNICATION_ERROR만 재시도(같은 목표 재전송이라 멱등) - CANCELED/FAULT는 재시도 안 함
        self.declare_parameter('rg2.communication_retry_count', 2)
        self.declare_parameter('rg2.communication_retry_backoff_s', 0.5)

        self.declare_parameter('servo.kp_xy', 1.2)
        self.declare_parameter('servo.kp_yaw', 1.0)
        self.declare_parameter('servo.v_max', 0.25)
        self.declare_parameter('servo.descend_speed', 0.10)
        self.declare_parameter('servo.eps_descend', 0.015)
        self.declare_parameter('servo.eps_grasp', 0.005)
        self.declare_parameter('servo.n_stable', 10)
        self.declare_parameter('servo.dt_latency', 0.05)
        self.declare_parameter('servo.timeout', 5.0)
        self.declare_parameter('servo.t_lost', 0.3)
        self.declare_parameter('servo.innov_low', 0.010)
        self.declare_parameter('servo.innov_high', 0.040)
        self.declare_parameter('servo.w_alpha', 0.3)
        self.declare_parameter('servo.z_close', 0.02)
        # z_close보다 작아야 한다 - 표면(z_gap=0)이 아닌 표면 위 이 마진에서 속도 0을 겨냥해 관성 여유를 둔다
        self.declare_parameter('servo.descend_stop_margin_m', 0.01)
        self.declare_parameter('servo.n_stable_z', 5)
        # z_gap 계산 목표를 순간 칼만 z 대신 최근 depth_valid raw z의 중앙값으로 안정화하는 창 크기
        self.declare_parameter('servo.z_stabilize_window', 5)
        self.declare_parameter('servo.diverge_n', 15)
        self.declare_parameter('servo.diverge_min_delta_m', 0.01)
        self.declare_parameter('servo.cov_threshold', 0.05)
        self.declare_parameter('servo.v_grasp_max', 0.05)
        self.declare_parameter('servo.n_stable_v', 5)
        self.declare_parameter('servo.v_tool_deadband_m_s', 0.03)
        # yaw_rate_max_deg_s는 vx/vy/v_max(m/s)와 단위가 다른(deg/s) 별도 상한이라 분리
        self.declare_parameter('servo.yaw_rate_max_deg_s', 30.0)
        self.declare_parameter('servo.eps_yaw_deg', 5.0)
        self.declare_parameter('servo.n_stable_yaw', 5)
        # base-frame grip_deg(vision) <-> TCP 방향 부호/오프셋 - 실기 캘리브레이션 전까지 항등(1.0/0.0),
        # hardware_ready 열기 전 반드시 확정할 것. yaw_offset_deg는 TCP 로컬 프레임에서
        # grip_deg=0에 해당하는 기준 벡터 방향(deg)을 뜻한다.
        self.declare_parameter('servo.yaw_sign', 1.0)
        self.declare_parameter('servo.yaw_offset_deg', 0.0)
        # yaw 발산 판정(should_abort) - xy의 diverge_n/diverge_min_delta_m과 대칭. 짐벌락 근방
        # 불안정성으로 인한 과회전을 물리적으로 제한하는 안전망.
        self.declare_parameter('servo.diverge_n_yaw', 15)
        self.declare_parameter('servo.diverge_min_delta_deg', 10.0)
        self.declare_parameter('servo.kalman_q_pos', 1e-4)
        self.declare_parameter('servo.kalman_q_vel', 1e-2)
        self.declare_parameter('servo.kalman_r_xy', 1e-4)
        self.declare_parameter('servo.kalman_r_z', 1e-4)
        self.declare_parameter('servo.kalman_p0_vel_reset', 1.0)
        self.declare_parameter('servo.command_validate_tolerance', 1e-6)

        self.declare_parameter('move.vel_deg_s', 30.0)
        self.declare_parameter('move.acc_deg_s2', 30.0)
        self.declare_parameter('move.blend_radius_mm', 0.0)
        self.declare_parameter('move.sync_type', 0)
        self.declare_parameter('move.dry_run_duration_s', 0.0)
        self.declare_parameter('move.poll_interval_s', 0.05)
        self.declare_parameter('move.timeout_s', 30.0)

        for name in NAMED_POSE_NAMES:
            _declare_double_array(self, f'named_poses.{name}', [])
        # dry-run에서만 유효 - hardware_enabled=true면 이 값과 무관하게 빈 pose는 항상 거부
        self.declare_parameter('dry_run.allow_unconfigured_named_poses', True)

        self.declare_parameter('handover_hold.pull_axis_index', -1)
        self.declare_parameter('handover_hold.pull_direction_sign', 1)
        self.declare_parameter('handover_hold.pull_force_threshold_n', 15.0)
        self.declare_parameter('handover_hold.poll_interval_s', 0.05)
        # DR_TOOL(1)은 DoosanDriver.get_tool_force가 호출 전에 거부한다
        self.declare_parameter('handover_hold.ref', 0)  # DR_BASE
        _declare_double_array(
            self, 'handover_hold.compliance_stiffness', [3000.0, 3000.0, 3000.0, 200.0, 200.0, 200.0])
        self.declare_parameter('handover_hold.compliance_transition_s', 0.4)
        # handover_hold 시작 전 수신된 오래된 힘 샘플로 당김을 오판하지 않기 위한 디바운스
        self.declare_parameter('handover_hold.force_sample_max_age_s', 0.5)
        self.declare_parameter('handover_hold.pull_confirm_samples', 3)

        # 외력 감지: DRFL 라이브러리에 ctypes로 직접 연결해 ROS2 executor와 무관한 독립
        # 쓰레드에서 고주기 폴링 - 관절별 절대 임계값 + 히스테리시스 방식이라 MOVING/STANDBY
        # 무관하게 항상 동작한다(이동 중 자세 변화로 오탐나는 delta 기반 판정 대신 사용).
        self.declare_parameter(
            'safety.external_torque.drfl_lib_path',
            '~/cobot_ws/install/dsr_hardware2/lib/libdsr_hardware2.so')
        self.declare_parameter('safety.external_torque.robot_ip', '192.168.1.100')
        self.declare_parameter('safety.external_torque.robot_port', 12345)
        self.declare_parameter('safety.external_torque.direct_poll_hz', 100.0)
        _declare_double_array(
            self, 'safety.external_torque.direct_threshold_nm',
            [15.0, 15.0, 12.0, 10.0, 10.0, 10.0])
        self.declare_parameter('safety.external_torque.direct_reset_below_count', 20)
        self.declare_parameter('safety.external_torque.stop_join_timeout_s', 2.0)
        self.declare_parameter('safety.fault_stop_mode', 1)  # DR_QSTOP: Quick stop Cat.2
        # diverging/timeout/tracking_lost 등 정상 회수·재시도 정지 전용(_run_rt_tracking) -
        # fault_stop_mode(QUICK)는 실제 안전 FAULT 전용으로 남겨둔다. STOP_TYPE_SLOW(2)는
        # QUICK 대비 완만해 반복 정지 시 충격이 적다. 0=QUICK_STO,1=QUICK,2=SLOW,3=HOLD(비상정지)
        self.declare_parameter('safety.recoverable_stop_mode', 2)
        self.declare_parameter('safety.state_poll_period_s', 0.1)
        # 진단용 - handover_hold 힘 임계값 실측 캘리브레이션 때만 켠다
        self.declare_parameter('safety.debug_log_tool_force', False)
        self.declare_parameter('gripper_poll_period_s', 0.5)
        self.declare_parameter('debug.publish_events', True)
        self.declare_parameter('debug.log_servo_decisions', False)
        self.declare_parameter('debug.log_safety_samples', False)
        self.declare_parameter('debug.log_gripper', False)

        # base_link -> link_6 TF는 이 노드가 방송하지 않는다(dsr_bringup2의
        # robot_state_publisher가 이미 글로벌 /tf에 방송 중 - 중복 방송 시 두 소스가 어긋나
        # vision_node TF lookup이 실패했었다). 아래 파라미터는 이제 _on_tf_broadcast_timer가
        # servo_pick TCP 캐시 갱신을 위해 GetCurrentPosx를 폴링하는 주기로만 쓰인다(이름은 이력용).
        self.declare_parameter('tf_broadcast.period_s', 0.02)
        self.declare_parameter('tf_broadcast.parent_frame_id', 'base_link')
        self.declare_parameter('tf_broadcast.child_frame_id', 'link_6')

        self.declare_parameter('doosan_driver.move_service_wait_timeout_s', 2.0)
        self.declare_parameter('doosan_driver.service_wait_timeout_s', 1.0)
        self.declare_parameter('doosan_driver.future_poll_interval_s', 0.01)
        self.declare_parameter('doosan_driver.future_wait_timeout_s', 2.0)
        self.declare_parameter('doosan_driver.compliance_future_wait_timeout_s', 3.0)
        # doosan-robot2 릴리스마다 dsr_controller2가 서비스/토픽 이름에 노드 이름을 붙이는지가
        # 달라 robot_id와 드라이버 사이 세그먼트를 파라미터로 분리 - 신 포크는 launch에서
        # local_params_file로 'dsr_controller2' 오버라이드.
        self.declare_parameter('doosan_driver.controller_name', '')

        # ToolTrack.pose는 base_link 절대좌표인데 ServoLoop는 TCP 기준 xy 오차로 가정하고
        # P 제어한다 - 이 좌표 변환이 검증되기 전까지는 게이트를 false로 막는다.
        self.declare_parameter('servo_pick.hardware_ready', False)
        self.declare_parameter('servo_pick.control_period_s', 0.01)
        self.declare_parameter('servo_pick.speedl_acc_trans_mm_s2', 100.0)
        self.declare_parameter('servo_pick.speedl_acc_rot_deg_s2', 30.0)
        # speedl(비-RT)은 명령이 끊겨도 스스로 멈추지 않는다 - SpeedlWatchdog가 이 시간 동안
        # pet()이 없으면 vel=0을 대신 발행한다.
        self.declare_parameter('servo_pick.watchdog_timeout_s', 0.2)
        self.declare_parameter('servo_pick.watchdog_poll_interval_s', 0.05)
        # ToolTrack이 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id
        self.declare_parameter('servo_pick.tool_track_frame_id', 'base_link')
        # 캐시(_tcp_pose_cache)가 이 값보다 오래되면 사용하지 않는다 - 칼만 ServoLoop는
        # dt가 벌어질수록 예측 불확실성이 누적돼 서보가 멈추므로 서보 루프에 맞춘 값이다.
        self.declare_parameter('servo_pick.tcp_pose_max_age_s', 0.2)
        # 그리퍼 폐합 확인 직후, VERIFY_GRASP 판정 전에 z를 이만큼(m) 들어올린다.
        # 0 이하면 들어올림을 건너뛴다.
        self.declare_parameter('servo_pick.lift_height_m', 0.05)
        # 들어올림 전용 속도 - descend_speed(표면 근접 제동거리에 맞춘 값)와 달리 위쪽에
        # 장애물이 없어 더 빠르게 둘 수 있다.
        self.declare_parameter('servo_pick.lift_speed_m_s', 0.15)
        # TCP 위치 피드백 정체 등으로 들어올림 단계가 무한 루프에 빠지지 않도록 하는 안전 타임아웃
        self.declare_parameter('servo_pick.lift_timeout_s', 5.0)

        # handover_safe 도착 후 hand_track을 따라 TCP->손 방향 위 offset_m 지점을 추종하다
        # 주먹이 확정되면 멈춘다(그리퍼 동작 없음, 이후 handover_hold가 당김을 기다림).
        self.declare_parameter('handover_servo.hardware_ready', True)
        self.declare_parameter('handover_servo.control_period_s', 0.01)
        self.declare_parameter('handover_servo.speedl_acc_trans_mm_s2', 100.0)
        self.declare_parameter('handover_servo.speedl_acc_rot_deg_s2', 30.0)
        self.declare_parameter('handover_servo.watchdog_timeout_s', 0.2)
        self.declare_parameter('handover_servo.watchdog_poll_interval_s', 0.05)
        # HandTrack이 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id
        self.declare_parameter('handover_servo.hand_track_frame_id', 'base_link')
        self.declare_parameter('handover_servo.tcp_pose_max_age_s', 0.2)
        # 사람에게 접근하는 동작이라 v_max는 servo_pick(0.25)보다 보수적으로 낮게 잡는다
        self.declare_parameter('handover_servo.kp_xy', 1.2)
        self.declare_parameter('handover_servo.kp_z', 1.2)
        self.declare_parameter('handover_servo.v_max', 0.15)
        self.declare_parameter('handover_servo.offset_m', 0.20)
        self.declare_parameter('handover_servo.t_lost_s', 0.3)
        # "주먹까지 계속 추종"을 위해 넉넉히 잡는다(0이면 비활성)
        self.declare_parameter('handover_servo.timeout_s', 60.0)
        self.declare_parameter('handover_servo.command_validate_tolerance', 1e-6)

        self.hardware_enabled = bool(self.get_parameter('hardware_enabled').value)
        self.safety_monitor = SafetyMonitor(self)
        self._named_poses = {name: [] for name in NAMED_POSE_NAMES}
        self._refresh_named_poses()

        self.action_callback_group = MutuallyExclusiveCallbackGroup()
        self.sensor_callback_group = ReentrantCallbackGroup()
        self.hardware_callback_group = ReentrantCallbackGroup()

        self.rg2_client = RG2Client(
            ip=self.get_parameter('rg2_ip').value,
            port=self.get_parameter('rg2_port').value,
            hardware_enabled=self.hardware_enabled,
            gripper=self.get_parameter('rg2_gripper').value,
            node=self)

        self.servo_loop = ServoLoop(
            kp_xy=self.get_parameter('servo.kp_xy').value,
            kp_yaw=self.get_parameter('servo.kp_yaw').value,
            v_max=self.get_parameter('servo.v_max').value,
            descend_speed=self.get_parameter('servo.descend_speed').value,
            eps_descend=self.get_parameter('servo.eps_descend').value,
            eps_grasp=self.get_parameter('servo.eps_grasp').value,
            n_stable=self.get_parameter('servo.n_stable').value,
            dt_latency=self.get_parameter('servo.dt_latency').value,
            timeout_s=self.get_parameter('servo.timeout').value,
            t_lost_s=self.get_parameter('servo.t_lost').value,
            innov_low=self.get_parameter('servo.innov_low').value,
            innov_high=self.get_parameter('servo.innov_high').value,
            w_alpha=self.get_parameter('servo.w_alpha').value,
            z_close=self.get_parameter('servo.z_close').value,
            n_stable_z=self.get_parameter('servo.n_stable_z').value,
            z_stabilize_window=self.get_parameter('servo.z_stabilize_window').value,
            diverge_n=self.get_parameter('servo.diverge_n').value,
            diverge_min_delta_m=self.get_parameter('servo.diverge_min_delta_m').value,
            cov_threshold=self.get_parameter('servo.cov_threshold').value,
            v_grasp_max=self.get_parameter('servo.v_grasp_max').value,
            n_stable_v=self.get_parameter('servo.n_stable_v').value,
            v_tool_deadband_m_s=self.get_parameter('servo.v_tool_deadband_m_s').value,
            yaw_rate_max_deg_s=self.get_parameter('servo.yaw_rate_max_deg_s').value,
            eps_yaw_deg=self.get_parameter('servo.eps_yaw_deg').value,
            n_stable_yaw=self.get_parameter('servo.n_stable_yaw').value,
            yaw_sign=self.get_parameter('servo.yaw_sign').value,
            yaw_offset_deg=self.get_parameter('servo.yaw_offset_deg').value,
            diverge_n_yaw=self.get_parameter('servo.diverge_n_yaw').value,
            diverge_min_delta_deg=self.get_parameter('servo.diverge_min_delta_deg').value,
            q_pos=self.get_parameter('servo.kalman_q_pos').value,
            q_vel=self.get_parameter('servo.kalman_q_vel').value,
            r_xy=self.get_parameter('servo.kalman_r_xy').value,
            r_z=self.get_parameter('servo.kalman_r_z').value,
            p0_vel_reset=self.get_parameter('servo.kalman_p0_vel_reset').value,
            # speedl 가속도 제한과 같은 값 재사용(단위만 변환) - 별도 파라미터로 중복 선언하면
            # 값을 바꿀 때 두 곳이 어긋날 수 있다.
            descend_accel_m_s2=(
                self.get_parameter('servo_pick.speedl_acc_trans_mm_s2').value / 1000.0),
            descend_stop_margin_m=self.get_parameter('servo.descend_stop_margin_m').value,
        )

        self.hand_servo_loop = HandServoLoop(
            kp_xy=self.get_parameter('handover_servo.kp_xy').value,
            kp_z=self.get_parameter('handover_servo.kp_z').value,
            v_max=self.get_parameter('handover_servo.v_max').value,
            offset_m=self.get_parameter('handover_servo.offset_m').value,
            t_lost_s=self.get_parameter('handover_servo.t_lost_s').value,
            timeout_s=self.get_parameter('handover_servo.timeout_s').value,
        )

        # DoosanDriver 초기화 실패 시 즉시 FAULT를 선언해야 하므로, 발행자를 먼저 만든다.
        self.pub_gripper_state = self.create_publisher(GripperState, '/gripper/state', 10)
        self.pub_fault = self.create_publisher(String, '/robot/fault', 10)
        self.pub_debug_events = self.create_publisher(String, '/debug/events', 10)

        self._init_doosan_driver()

        # goal 수락 경쟁(TOCTOU) 방지 - execute_callback 종료 시(finally) 예약 해제
        self._goal_lock = threading.Lock()
        self._goal_reserved = False
        self._handlers = {
            'move_named': self._execute_move_named,
            'release_and_retry': self._execute_release_and_retry,
            'servo_pick': self._execute_servo_pick,
            'handover_hold': self._execute_handover_hold,
            'handover_approach': self._execute_handover_approach,
        }

        self._action_server = ActionServer(
            self, RobotTask, 'robot_task',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.action_callback_group)

        # _on_tf_broadcast_timer가 채우고 _get_current_tcp_posx()가 읽는 TCP 위치 캐시 -
        # ToolTrack 콜백에서 매번 동기 서비스 호출을 하지 않기 위함.
        self._tcp_pose_cache = None
        # 응답이 타이머 주기보다 오래 걸릴 수 있어 겹쳐서 새 요청을 보내지 않게 막는 가드
        self._tcp_pose_request_in_flight = False
        self._tcp_tracking_active = False
        # servo_pick이 RG2 close를 백그라운드 스레드로 돌리는 동안 쓰는 상태
        self._servo_pick_close_thread = None
        self._servo_pick_close_success = None
        self._gripper_timer = self.create_timer(
            self.get_parameter('gripper_poll_period_s').value,
            self._on_gripper_timer, callback_group=self.sensor_callback_group)
        self._tf_broadcast_timer = self.create_timer(
            self.get_parameter('tf_broadcast.period_s').value,
            self._on_tf_broadcast_timer, callback_group=self.sensor_callback_group)
        self._state_poll_timer = self.create_timer(
            self.get_parameter('safety.state_poll_period_s').value,
            self._on_state_poll_timer, callback_group=self.sensor_callback_group)

        self.recover_srv = self.create_service(
            Trigger, '/robot/recover', self._on_recover, callback_group=self.sensor_callback_group)

    @property
    def safety_state(self):
        return self.safety_monitor.state

    @safety_state.setter
    def safety_state(self, value):
        self.safety_monitor.state = value

    def _checkpoint_event(
            self, phase, checkpoint_id, status, message, data=None,
            *, throttle_s=None, log=False):
        """파이프라인 점검.md의 Phase 체크리스트에 대응하는 이벤트를 발행한다."""
        now = time.monotonic()
        key = (checkpoint_id, status)
        if throttle_s is not None:
            last = getattr(self, '_checkpoint_event_last', {}).get(key, 0.0)
            if now - last < throttle_s:
                return
            if not hasattr(self, '_checkpoint_event_last'):
                self._checkpoint_event_last = {}
            self._checkpoint_event_last[key] = now
        payload = {
            'phase': phase,
            'checkpoint_id': checkpoint_id,
            'status': status,
            'message': message,
            'data': data or {},
            'node': self.get_name(),
            'stamp_monotonic': now,
        }
        if bool(self.get_parameter('debug.publish_events').value):
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub_debug_events.publish(msg)
        if log:
            text = f'[CHECKPOINT][{phase}/{checkpoint_id}] status={status} message={message}'
            if status == 'FAIL':
                self.get_logger().error(text)
            else:
                self.get_logger().info(text)

    @property
    def _latest_robot_state(self):
        return self.safety_monitor.latest_robot_state

    @_latest_robot_state.setter
    def _latest_robot_state(self, value):
        self.safety_monitor.latest_robot_state = value

    @property
    def _last_fault_reason(self):
        return self.safety_monitor.last_fault_reason

    @_last_fault_reason.setter
    def _last_fault_reason(self, value):
        self.safety_monitor.last_fault_reason = value

    def _refresh_named_poses(self):
        for name in NAMED_POSE_NAMES:
            param_name = f'named_poses.{name}'
            alternative = Parameter(param_name, Parameter.Type.DOUBLE_ARRAY, [])
            value = self.get_parameter_or(param_name, alternative).value
            self._named_poses[name] = list(value) if value else []

    def _init_doosan_driver(self):
        """생성 실패 시 즉시 safety_state=FAULT를 선언해 이후 모든 Goal을 거부한다 -
        하드웨어 경계가 없는 상태로 조용히 dry_run처럼 동작하지 않는다."""
        self._doosan = None
        self._drfl_force_monitor = None
        if not self.hardware_enabled:
            return
        try:
            self._doosan = DoosanDriver(self)
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            self.safety_state = SafetyState.FAULT
            fault_msg = String()
            fault_msg.data = f'{FaultPrefix.FAULT}DoosanDriver 초기화 실패: {exc}'
            self.pub_fault.publish(fault_msg)
            return
        self._init_drfl_force_monitor()

    def _init_drfl_force_monitor(self):
        """보조 외력 감지 레이어 - 필수 경로가 아니므로 연결 실패해도 FAULT를 선언하지 않는다."""
        try:
            thresholds = self.get_parameter(
                'safety.external_torque.direct_threshold_nm').value
            self._drfl_force_monitor = DrflForceMonitor(
                lib_path=os.path.expanduser(
                    self.get_parameter('safety.external_torque.drfl_lib_path').value),
                robot_ip=self.get_parameter('safety.external_torque.robot_ip').value,
                robot_port=int(self.get_parameter('safety.external_torque.robot_port').value),
                thresholds_nm=thresholds,
                on_triggered=self._on_drfl_force_triggered,
                poll_hz=self.get_parameter('safety.external_torque.direct_poll_hz').value,
                reset_below_count=self.get_parameter(
                    'safety.external_torque.direct_reset_below_count').value,
                stop_join_timeout_s=self.get_parameter(
                    'safety.external_torque.stop_join_timeout_s').value,
            )
            self._drfl_force_monitor.start()
        except Exception as exc:
            self.get_logger().error(
                f'DRFL 직접 외력 감지 초기화 실패 - 이 보조 레이어만 비활성화됩니다: {exc}')
            self._drfl_force_monitor = None

    def _suspend_drfl_force_monitor(self):
        """handover_hold 컴플라이언스 구간처럼 접촉력이 기대되는 동안 FAULT 선언을 일시 정지 -
        이 구간의 안전 판단은 handover_hold의 당김 감지(_is_pull_detected)가 대신한다."""
        if getattr(self, '_drfl_force_monitor', None) is not None:
            self._drfl_force_monitor.suspend()

    def _resume_drfl_force_monitor(self):
        if getattr(self, '_drfl_force_monitor', None) is not None:
            self._drfl_force_monitor.resume()

    def _on_drfl_force_triggered(self, joint_index, value, threshold):
        """DrflForceMonitor의 백그라운드 쓰레드에서 호출된다(ROS2 executor 쓰레드가 아니다) -
        declare_fault/publish는 어느 쓰레드에서 불러도 안전하다."""
        reason = (
            f'{FaultPrefix.FAULT}예상하지 못한 외력이 감지되었습니다(이동 중 포함 직접 감지) '
            f'(joint={joint_index + 1}, 값={value:.1f} Nm, 기준={threshold:.1f} Nm).')
        self.get_logger().error(reason)
        self.safety_monitor.declare_fault(reason)

    def destroy_node(self):
        if getattr(self, '_drfl_force_monitor', None) is not None:
            self._drfl_force_monitor.stop()
        super().destroy_node()

    # ---- goal 수락/취소 ----

    def _goal_callback(self, goal_request):
        checkpoint = _GOAL_SENT_CHECKPOINTS.get(
            (goal_request.task_type, goal_request.named_target))

        def _publish_reject(message):
            if checkpoint is not None:
                phase, checkpoint_id = checkpoint
                self._checkpoint_event(phase, checkpoint_id, 'FAIL', message,
                                        {'task_type': goal_request.task_type})

        if self.safety_state != SafetyState.NORMAL:
            self.get_logger().warn(f'Goal 거부 - safety_state={self.safety_state}')
            _publish_reject(f'안전상태가 NORMAL이 아니어서 goal을 거부했습니다({self.safety_state}).')
            return GoalResponse.REJECT
        if goal_request.task_type not in self._handlers:
            self.get_logger().warn(f'Goal 거부 - 알 수 없는 task_type: {goal_request.task_type}')
            _publish_reject(f'알 수 없는 task_type입니다: {goal_request.task_type}')
            return GoalResponse.REJECT
        if (goal_request.task_type == 'servo_pick' and self.hardware_enabled
                and not self.get_parameter('servo_pick.hardware_ready').value):
            self.get_logger().warn(
                'Goal 거부 - servo_pick.hardware_ready=false (ToolTrack 좌표 변환 미검증)')
            return GoalResponse.REJECT
        if (goal_request.task_type == 'handover_approach' and self.hardware_enabled
                and not self.get_parameter('handover_servo.hardware_ready').value):
            self.get_logger().warn(
                'Goal 거부 - handover_servo.hardware_ready=false (hand_track 좌표 변환 미검증)')
            _publish_reject('handover_servo.hardware_ready=false라 접근 goal을 거부했습니다.')
            return GoalResponse.REJECT
        # goal 수락 경쟁(TOCTOU) 방지: 락 안에서 원자적으로 하나만 예약한다.
        with self._goal_lock:
            if self._goal_reserved:
                self.get_logger().warn('Goal 거부 - 이미 실행 중(또는 취소 처리 중)인 goal이 있습니다.')
                _publish_reject('이미 실행 중인 goal이 있어 새 goal을 거부했습니다.')
                return GoalResponse.REJECT
            self._goal_reserved = True
        if checkpoint is not None:
            phase, checkpoint_id = checkpoint
            self._checkpoint_event(
                phase, checkpoint_id, 'PASS', 'goal이 수락되었습니다.',
                {'task_type': goal_request.task_type})
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    # ---- fault / robot state polling ----

    def _read_robot_state(self):
        return self.safety_monitor.read_robot_state()

    def _check_fault(self, robot_state):
        return self.safety_monitor.check_fault(robot_state)

    @staticmethod
    def _classify_fault_level(reason):
        return SafetyMonitor.classify_fault_level(reason)

    def _declare_fault(self, reason):
        self.safety_monitor.declare_fault(reason)

    def _on_state_poll_timer(self):
        state = self._read_robot_state()
        if state is None:
            return
        self._latest_robot_state = state
        if bool(self.get_parameter('safety.debug_log_tool_force').value):
            tool_force = state.get('tool_force') if isinstance(state, dict) else None
            if tool_force:
                self.get_logger().info(
                    f'[tool_force debug] fx={tool_force[0]:.1f} fy={tool_force[1]:.1f} '
                    f'fz={tool_force[2]:.1f} mx={tool_force[3]:.1f} my={tool_force[4]:.1f} '
                    f'mz={tool_force[5]:.1f} (N, Nm / DR_BASE 기준)',
                    throttle_duration_sec=1.0)
        if bool(self.get_parameter('debug.log_safety_samples').value):
            self.get_logger().info(
                f"[SAFETY_SAMPLE] robot_state={state.get('robot_state')} "
                f"tool_force={state.get('tool_force')} ext_torque={state.get('ext_torque')}",
                throttle_duration_sec=1.0)
        reason = self._check_fault(state)
        if reason is not None and reason != self._last_fault_reason:
            self._declare_fault(reason)

    def _on_gripper_timer(self):
        width_mm, grip_detected = self.rg2_client.get_state()
        msg = GripperState()
        msg.width_mm = width_mm
        msg.grip_detected = grip_detected
        self.pub_gripper_state.publish(msg)
        if bool(self.get_parameter('debug.log_gripper').value):
            self.get_logger().info(
                f'[GRIPPER_SAMPLE] width_mm={width_mm} grip_detected={bool(grip_detected)}',
                throttle_duration_sec=1.0)

    def _on_tf_broadcast_timer(self):
        """GetCurrentPosx를 폴링해 servo_pick의 TCP 위치 캐시(_tcp_pose_cache)를 채운다
        (칼만 ServoLoop.step()이 매 RT tick 읽는 값). base_link -> link_6 TF는 방송하지
        않는다 - dsr_bringup2의 robot_state_publisher가 이미 글로벌 /tf에 방송 중이라
        중복 방송하면 두 소스가 어긋난다(이름은 이력용)."""
        if not self.hardware_enabled or self._doosan is None:
            return
        if self._tcp_pose_request_in_flight:
            return  # 이전 GetCurrentPosx 응답 대기 중 - 겹쳐서 새로 호출하지 않는다
        self._tcp_pose_request_in_flight = True
        try:
            pos6 = self._doosan.get_current_posx(ref=0)
        finally:
            self._tcp_pose_request_in_flight = False
        if pos6 is None:
            return
        if self._tcp_tracking_active and self.safety_state == SafetyState.NORMAL:
            self._tcp_pose_cache = {'pos6': pos6, 'received_at': time.monotonic()}

    # ---- /robot/recover ----

    def _on_recover(self, request, response):
        return self.safety_monitor.recover(request, response)

    # ---- action dispatch ----

    def _execute_callback(self, goal_handle):
        try:
            task_type = goal_handle.request.task_type
            handler = self._handlers.get(task_type)
            if handler is None:
                goal_handle.abort()
                result = RobotTask.Result()
                result.success = False
                result.message = f'unknown task_type: {task_type}'
                return result
            return handler(goal_handle)
        finally:
            with self._goal_lock:
                self._goal_reserved = False


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()