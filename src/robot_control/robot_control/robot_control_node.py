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

# _goal_callback의 (task_type, named_target) -> (phase, checkpoint_id).
# servo_pick/release_and_retry 및 named_target이 front/up/down인 수동 이동은
# 파이프라인 점검.md 체크리스트 항목이 아니므로 체크포인트를 발행하지 않는다.
_GOAL_SENT_CHECKPOINTS = {
    ('move_named', 'watch'): ('B', 'move_watch_goal_sent'),
    ('move_named', 'handover_safe'): ('F', 'handover_safe_goal_sent'),
    ('move_named', 'home'): ('J', 'home_goal_sent'),
    ('handover_approach', ''): ('H', 'handover_approach_goal_sent'),
    ('handover_hold', ''): ('I', 'handover_hold_goal_sent'),
}


def _declare_double_array(node, name, default):
    if not default:
        # rclpy Humble 버그: declare_parameter에 빈 리스트([])를 기본값으로 주면
        # Parameter.Type.from_parameter_value([])가 항상 BYTE_ARRAY로 추론한다
        # (all(...)이 빈 시퀀스에서 True이기 때문) - ParameterDescriptor로 명시한
        # DOUBLE_ARRAY 타입이 그 추론 결과에 덮어써진다. Parameter.Type을 직접
        # 넘겨 타입 추론 자체를 건너뛴다 - 대신 override가 없으면 파라미터가
        # 미초기화 상태로 남으므로, 읽는 쪽(_refresh_named_poses)에서
        # get_parameter_or로 빈 배열 기본값을 되돌려줘야 한다.
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
        # RG2 open/close 명령이 busy=0(완료)이 될 때까지 기다리는 통신 타임아웃/폴링
        # 주기 - 하드웨어 캘리브레이션 값이 아니라 통신 타이밍 설정이다.
        self.declare_parameter('rg2.command_timeout_s', 5.0)
        self.declare_parameter('rg2.poll_interval_s', 0.05)
        # open() 완료 후 최종 폭이 "최대 폭에 도달했다"고 참고로 판단할 때 허용할
        # 오차(mm) - 실측으로 확정된 값이 아니라 통신/기구적 오차를 감안한 여유값이다.
        self.declare_parameter('rg2.open_width_tolerance_mm', 2.0)
        # Modbus 소켓 타임아웃 - 이전에는 1초로 하드코딩돼 있어 네트워크가 잠깐
        # 느려지기만 해도 COMMUNICATION_ERROR로 이어졌다.
        self.declare_parameter('rg2.connect_timeout_s', 2.0)
        # COMMUNICATION_ERROR에 한해서만 자동 재시도한다(같은 목표를 다시 보내는
        # 멱등한 재시도라 안전함) - CANCELED/FAULT 등 다른 상태는 재시도하지 않는다
        # (RG2Client._run_command_with_retry 참고).
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
        # innovation(예측-관측 잔차) 기반 피드포워드 가중치 w 및 폐합/발산 판정 임계값
        # (servo_loop.py ServoLoop 참고) - 이전에는 코드 상수였으나, kp_xy 등 다른
        # 서보 게인과 마찬가지로 실기 튜닝 대상이라 ROS 파라미터로 승격했다.
        self.declare_parameter('servo.innov_low', 0.010)
        self.declare_parameter('servo.innov_high', 0.040)
        self.declare_parameter('servo.w_alpha', 0.3)
        self.declare_parameter('servo.z_close', 0.02)
        self.declare_parameter('servo.n_stable_z', 5)
        self.declare_parameter('servo.diverge_n', 15)
        self.declare_parameter('servo.cov_threshold', 0.05)
        # ServoLoop 내부 KalmanXYZV(kalman.py)로 그대로 전달되는 필터 노이즈
        # 파라미터 - 위와 같은 이유로 코드 상수에서 ROS 파라미터로 승격했다.
        self.declare_parameter('servo.kalman_q_pos', 1e-4)
        self.declare_parameter('servo.kalman_q_vel', 1e-2)
        self.declare_parameter('servo.kalman_r_xy', 1e-4)
        self.declare_parameter('servo.kalman_r_z', 1e-4)
        self.declare_parameter('servo.kalman_p0_vel_reset', 1.0)
        # _validate_servo_command(task_executor.py)의 부동소수점 비교 허용오차 -
        # 물리량이 아닌 수치 오차 마진이지만, 위 값들과 함께 한 곳에서 보이도록
        # 승격했다.
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
        # hardware_enabled=false(dry-run)에서는 실측 관절값이 없는 named pose도
        # 상태 흐름 시험을 위해 이동을 허용한다. hardware_enabled=true에서는 이 값과
        # 무관하게 빈 pose를 절대 허용하지 않는다 (_call_move_service 참고).
        self.declare_parameter('dry_run.allow_unconfigured_named_poses', True)

        self.declare_parameter('handover_hold.pull_axis_index', -1)
        self.declare_parameter('handover_hold.pull_direction_sign', 1)
        self.declare_parameter('handover_hold.pull_force_threshold_n', 15.0)
        self.declare_parameter('handover_hold.poll_interval_s', 0.05)
        # GetToolForce.srv는 DR_BASE(0)/DR_TOOL(1)/DR_WORLD(2)를 정의하지만, 이 노드는
        # DR_TOOL(1)을 허용하지 않는다 (DoosanDriver.get_tool_force가 호출 전에 거부).
        self.declare_parameter('handover_hold.ref', 0)  # DR_BASE
        _declare_double_array(
            self, 'handover_hold.compliance_stiffness', [3000.0, 3000.0, 3000.0, 200.0, 200.0, 200.0])
        self.declare_parameter('handover_hold.compliance_transition_s', 0.4)
        # handover_hold 시작 이전에 수신된 오래된 힘 샘플로 당김을 오판하지 않도록,
        # 샘플의 최대 허용 나이(초)와 연속 확인 횟수를 파라미터로 둔다 (실제 축/임계값과
        # 달리 이 값들은 타이밍/디바운스 설정이라 임의 하드웨어 값을 추측하는 것이 아니다).
        self.declare_parameter('handover_hold.force_sample_max_age_s', 0.5)
        self.declare_parameter('handover_hold.pull_confirm_samples', 3)

        # 외력 감지: dsr_msgs2 ROS 서비스가 아니라 DRFL 라이브러리에 ctypes로 직접
        # 연결해 ROS2 executor와 무관한 독립 쓰레드에서 고주기(기본 100Hz)로 폴링한다
        # (rokey_proj_01의 force_monitor_node.py와 동일 접근, 2026-07-06 도입).
        # 관절별 절대 임계값 + 히스테리시스(reset_below_count) 방식이라, MOVING
        # 중이든 STANDBY든 상관없이 항상 동작한다 - "최근 평균 대비 변화량(delta)"
        # 으로 판단하던 이전 방식은 정지 상태에서만 유효하고 이동 중엔 자세 변화
        # 자체로 오탐이 나서(2026-07-06 확인) 이 방식으로 완전히 대체했다.
        self.declare_parameter(
            'safety.external_torque.drfl_lib_path',
            '/home/youngjin/cobot_ws/install/dsr_hardware2/lib/libdsr_hardware2.so')
        self.declare_parameter('safety.external_torque.robot_ip', '192.168.1.100')
        self.declare_parameter('safety.external_torque.robot_port', 12345)
        self.declare_parameter('safety.external_torque.direct_poll_hz', 100.0)
        _declare_double_array(
            self, 'safety.external_torque.direct_threshold_nm',
            [15.0, 15.0, 12.0, 10.0, 10.0, 10.0])
        self.declare_parameter('safety.external_torque.direct_reset_below_count', 20)
        # DrflForceMonitor.stop()이 폴링 스레드 종료를 기다리는 타임아웃 - 내부 종료
        # 안전 마진이었으나 다른 external_torque.* 값들과 함께 보이도록 승격했다.
        self.declare_parameter('safety.external_torque.stop_join_timeout_s', 2.0)
        self.declare_parameter('safety.fault_stop_mode', 1)  # DR_QSTOP: Quick stop Cat.2
        self.declare_parameter('safety.state_poll_period_s', 0.1)
        # 진단용: true면 상태 폴링마다 tool_force를 1초 간격으로 로그에 남긴다.
        # handover_hold.pull_axis_index/pull_direction_sign/pull_force_threshold_n을
        # 실측으로 잡을 때만 켠다 - 기본 실행에서는 로그 스팸을 피하기 위해 false.
        self.declare_parameter('safety.debug_log_tool_force', False)
        self.declare_parameter('gripper_poll_period_s', 0.5)
        # DEBUG_LOG: 실기 디버깅용 구조화 이벤트. 안정화 후 GUI/로그 정책 확정 시 제거 가능.
        self.declare_parameter('debug.publish_events', True)
        self.declare_parameter('debug.log_servo_decisions', False)
        self.declare_parameter('debug.log_safety_samples', False)
        self.declare_parameter('debug.log_gripper', False)

        # base_link -> link_6 TF는 이 노드가 방송하지 않는다. dsr_bringup2가 띄우는
        # /dsr01/robot_state_publisher가 /dsr01/joint_states(실측 100Hz, ros2_control)로
        # 순기구학 계산해 이미 같은 프레임을 글로벌 /tf에 방송하고 있다(remap_tf 기본값
        # false라 네임스페이스 없이 글로벌 /tf로 나간다 - dsr_bringup2_rviz.launch.py 확인).
        # 2026-07-08 실기 검증 중 이 노드가 GetCurrentPosx로 별도 방송하던 버전은
        # `ros2 topic info /tf --verbose`로 "/tf에 발행자가 2개(robot_state_publisher,
        # robot_control)"임을 확인했고, tf2_echo에서 두 소스가 약 230mm씩 어긋난 값을
        # 번갈아 내놓는 것도 확인됨 - vision_node의 TF lookup이 계속 "extrapolation
        # into the future"로 실패하던 근본 원인이었다(폴링 주기를 50Hz->10Hz로 낮춰도
        # 안 고쳐졌던 이유). 이제 아래 파라미터들은 _on_tf_broadcast_timer가 servo_pick
        # TCP 캐시 갱신을 위해 GetCurrentPosx를 폴링하는 주기로만 쓰인다(이름은 과거
        # 이력 그대로 남겨둠).
        self.declare_parameter('tf_broadcast.period_s', 0.02)
        self.declare_parameter('tf_broadcast.parent_frame_id', 'base_link')
        self.declare_parameter('tf_broadcast.child_frame_id', 'link_6')

        # DoosanDriver(doosan_driver.py)가 dsr_msgs2 서비스를 부를 때 쓰는 통신
        # 타임아웃/폴링 간격 - rg2.command_timeout_s 등과 같은 성격의 통신 타이밍
        # 상수였으나 한 곳에서 보이도록 ROS 파라미터로 승격했다.
        self.declare_parameter('doosan_driver.move_service_wait_timeout_s', 2.0)
        self.declare_parameter('doosan_driver.service_wait_timeout_s', 1.0)
        self.declare_parameter('doosan_driver.future_poll_interval_s', 0.01)
        self.declare_parameter('doosan_driver.future_wait_timeout_s', 2.0)
        self.declare_parameter('doosan_driver.compliance_future_wait_timeout_s', 3.0)

        # servo_pick 실제 하드웨어 실행을 위한 별도 게이트. hardware_enabled=true여도
        # 이 값이 false면 servo_pick Goal 자체를 거부한다 (기본값 false).
        # 이유: 현재 ToolTrack.pose는 base_link 절대좌표로 정의되어 있는데
        # (handover_interfaces/msg/ToolTrack.msg), ServoLoop는 이를 TCP(그리퍼) 기준
        # xy 오차로 가정하고 P 제어를 수행한다 (servo_loop.py 상단 주석 참고). 이 좌표
        # 변환이 실제로 구현·검증되기 전까지는 실제 속도 명령을 로봇에 보내면 안 된다.
        self.declare_parameter('servo_pick.hardware_ready', False)
        self.declare_parameter('servo_pick.control_period_s', 0.01)
        self.declare_parameter('servo_pick.speedl_acc_trans_mm_s2', 100.0)
        self.declare_parameter('servo_pick.speedl_acc_rot_deg_s2', 30.0)
        # speedl(비-RT)은 명령이 끊겨도 스스로 멈추지 않는다(2026-07-07
        # probe_speedl_stream.py로 실측 확인) - SpeedlWatchdog가 이 시간 동안
        # pet()이 없으면 vel=0을 대신 발행한다. 단일 정지 명령으로 충분함도
        # 같은 실측으로 확인됨.
        self.declare_parameter('servo_pick.watchdog_timeout_s', 0.2)
        # SpeedlWatchdog(speedl_watchdog.py)이 pet() 유무를 확인하는 내부 폴링 주기 -
        # 통신 타이밍 상수였으나 다른 servo_pick.* 값들과 함께 한 곳에서 보이도록
        # ROS 파라미터로 승격했다.
        self.declare_parameter('servo_pick.watchdog_poll_interval_s', 0.05)
        # ToolTrack이 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id.
        # TF 변환이 구현되지 않았으므로 다른 frame_id는 거부한다 (_compute_tool_track_tcp_offset).
        self.declare_parameter('servo_pick.tool_track_frame_id', 'base_link')
        # TCP 위치 캐시 샘플의 나이(초)가 이 값보다 크면 오래됐다고 보고 사용하지
        # 않는다 (서비스 왕복 시간이 아니라 _tcp_pose_cache 샘플 자체의 나이를
        # 뜻한다 - _on_tf_broadcast_timer/_get_current_tcp_posx 참고). 하드웨어
        # 캘리브레이션 값이 아니라 서보 제어 루프에 맞는 통신 타이밍 설정이다.
        # 2026-07-08: 칼만 ServoLoop는 dt가 벌어질수록 예측 불확실성(Q)이 누적되고,
        # 캐시가 이 값보다 오래 묵으면 매 tick 명령을 계산할 최신 TCP 위치 자체가
        # 없어 서보가 아예 멈춘다. 이전에는 이 캐시를 tcp_pose_refresh_period_s(20Hz)로
        # 별도 폴링했는데, tf_broadcast.period_s(50Hz) 폴링과 같은 GetCurrentPosx
        # 서비스를 이중으로 때려 스레드 고갈 및 TF 정지를 유발했다(실기 확인) - 이제
        # TF 방송용 폴링 결과에 캐시 갱신을 얹어 폴링 스트림을 하나로 합쳤다
        # (_on_tf_broadcast_timer 참고, tcp_pose_refresh_period_s 파라미터는 제거됨).
        self.declare_parameter('servo_pick.tcp_pose_max_age_s', 0.2)

        # handover_servo: handover_safe 도착 후 /vision/hand_track(작업자 손 위치, 연속
        # 스트림)을 따라 TCP->손 방향 위 offset_m 지점을 계속 추종하다 주먹이 확정되면
        # 멈춘다(그리퍼 동작 없음 - 이후 handover_hold가 당김을 기다린다). servo_pick과
        # 동일한 speedl 기반 RT 서보 루프(_run_rt_tracking)를 그대로 재사용한다.
        # hardware_ready는 servo_pick.hardware_ready와 같은 이유로 기본 false다:
        # hand_track(vision_node._track_hand)이 아직 연속 발행/주먹 판정을 지원하지
        # 않아 frame_id/orientation/fist 의미가 검증되지 않았다 - 확정 전까지 실제
        # 속도 명령 발행을 금지한다.
        self.declare_parameter('handover_servo.hardware_ready', False)
        self.declare_parameter('handover_servo.control_period_s', 0.01)
        self.declare_parameter('handover_servo.speedl_acc_trans_mm_s2', 100.0)
        self.declare_parameter('handover_servo.speedl_acc_rot_deg_s2', 30.0)
        # speedl watchdog - servo_pick과 동일한 이유(명령 스트림이 끊겨도 로봇이
        # 스스로 멈추지 않아 데드맨 스위치가 필요함).
        self.declare_parameter('handover_servo.watchdog_timeout_s', 0.2)
        self.declare_parameter('handover_servo.watchdog_poll_interval_s', 0.05)
        # HandTrack이 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id.
        self.declare_parameter('handover_servo.hand_track_frame_id', 'base_link')
        # TCP 위치 캐시 샘플의 나이(초) 상한 - servo_pick.tcp_pose_max_age_s와 동일한
        # 이유(_get_current_tcp_posx 참고).
        self.declare_parameter('handover_servo.tcp_pose_max_age_s', 0.2)
        # 수평/수직 P 게인 - 사람에게 접근하는 동작이라 v_max는 servo_pick(0.25)보다
        # 보수적으로 낮게 잡는다.
        self.declare_parameter('handover_servo.kp_xy', 1.2)
        self.declare_parameter('handover_servo.kp_z', 1.2)
        self.declare_parameter('handover_servo.v_max', 0.15)
        # 사용자 확정값(TCP->손 방향 위, 손 앞 20cm 지점에서 추종을 멈춘다).
        self.declare_parameter('handover_servo.offset_m', 0.20)
        # 손 유실 판정 시간 - servo_pick.t_lost(0.3s)와 동일한 성격.
        self.declare_parameter('handover_servo.t_lost_s', 0.3)
        # 타임아웃 - 사용자는 "주먹까지 계속 추종"을 원하므로 넉넉히 잡는다(0이면 비활성).
        self.declare_parameter('handover_servo.timeout_s', 60.0)
        # _validate_handover_servo_command(task_executor.py)의 부동소수점 비교 허용오차.
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
            diverge_n=self.get_parameter('servo.diverge_n').value,
            cov_threshold=self.get_parameter('servo.cov_threshold').value,
            q_pos=self.get_parameter('servo.kalman_q_pos').value,
            q_vel=self.get_parameter('servo.kalman_q_vel').value,
            r_xy=self.get_parameter('servo.kalman_r_xy').value,
            r_z=self.get_parameter('servo.kalman_r_z').value,
            p0_vel_reset=self.get_parameter('servo.kalman_p0_vel_reset').value,
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

        # goal 수락 경쟁(TOCTOU) 방지용: goal_callback 안에서 락을 잡고 원자적으로
        # 하나의 goal만 예약한다. execute_callback 종료 시(finally) 예약을 해제한다.
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

        # TCP 위치 캐시 - _on_tf_broadcast_timer가 조회한 GetCurrentPosx 결과를
        # 채운다(아래 타이머 설명 참고). _get_current_tcp_posx()는 이 캐시만 읽는다
        # (ToolTrack 콜백에서 매번 동기 서비스 호출을 하지 않기 위함).
        self._tcp_pose_cache = None
        # _on_tf_broadcast_timer의 GetCurrentPosx 호출을 감싸는 in-flight 가드 -
        # 응답이 타이머 주기보다 오래 걸릴 수 있어 겹쳐서 새 요청을 보내지 않게 막는다
        # (과거에는 이 폴링과 별도 타이머가 각자 요청을 겹쳐 보내 스레드 고갈을
        # 유발했다).
        self._tcp_pose_request_in_flight = False
        # servo_pick 또는 handover_approach가 실제로 실행 중일 때만 TCP 위치를
        # 캐시에 반영한다 - 불필요한 상태 갱신을 피하기 위함이다.
        self._tcp_tracking_active = False
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
        """hardware_enabled=true일 때 DoosanDriver를 생성한다.

        생성에 실패하면(예: dsr_msgs2 미설치) 즉시 safety_state=FAULT를 선언해
        goal_callback이 이후의 모든 Goal을 거부하도록 한다 (하드웨어 경계가 없는
        상태로 조용히 dry_run처럼 동작하지 않는다).
        """
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
        """MOVING 중에도 동작하는 보조 외력 감지 레이어를 시작한다 (drfl_force_monitor
        참고). 이 레이어는 안전상 "있으면 더 좋은" 보조 수단이지 필수 경로가 아니므로,
        연결 실패해도 FAULT를 선언하지 않는다 - 기존 STANDBY delta 체크와 두산 자체
        안전시스템은 이것과 무관하게 그대로 동작한다."""
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

    def _on_drfl_force_triggered(self, joint_index, value, threshold):
        """DrflForceMonitor의 백그라운드 쓰레드에서 직접 호출된다 (ROS2 executor
        쓰레드가 아니다). declare_fault/publish 호출은 doosan_driver의 다른 동기
        호출들과 마찬가지로 어느 쓰레드에서 불러도 안전하다 - _wait_for_future가
        executor를 spin하지 않고 단순 폴링만 하므로 서로 경합하지 않는다."""
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
            # 당김 감지(handover_hold.pull_axis_index/pull_direction_sign/
            # pull_force_threshold_n) 실측 캘리브레이션 전용 - 1초에 한 번만 남긴다.
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
        """GetCurrentPosx를 폴링해 servo_pick의 TCP 위치 캐시(_tcp_pose_cache)를
        채운다(칼만 ServoLoop.step()이 매 RT tick 읽는 값). dry-run(hardware_enabled=false)
        에서는 실제 자세가 없으므로 아무것도 하지 않는다.

        base_link -> link_6 TF는 이 함수가 더 이상 방송하지 않는다. 2026-07-08 실기
        검증 중 `ros2 topic info /tf --verbose`로 이 노드와 dsr_bringup2의
        /dsr01/robot_state_publisher가 똑같은 base_link->link_6를 동시에 글로벌 /tf에
        방송하고 있음을 확인했고(remap_tf 기본값 false), tf2_echo에서 두 소스가 약
        230mm씩 어긋난 값을 번갈아 내놓는 것도 확인했다 - vision_node의 TF lookup이
        "extrapolation into the future"로 계속 실패하던 근본 원인이었다(GetCurrentPosx
        폴링 주기를 50Hz->10Hz로 낮춰도 안 고쳐졌던 이유이기도 하다). robot_state_publisher
        쪽은 /dsr01/joint_states(ros2_control, 실측 100Hz)로 순기구학 계산하므로 이
        노드가 GetCurrentPosx로 별도 방송할 필요가 없다 - 그래서 TF 방송 코드를
        제거하고 TCP 캐시 갱신 폴링만 남겼다.

        (과거 이력: 원래 이 타이머(50Hz, TF 전용)와 _on_tcp_pose_refresh_timer(20Hz,
        TCP 캐시 전용)가 각자 in-flight 가드를 갖고 독립적으로 GetCurrentPosx를
        호출해 스레드 고갈을 유발했던 것을 이 타이머 하나로 합쳤다 - in-flight 가드는
        여전히 필요하다(호출 자체가 timer period보다 오래 걸릴 수 있으므로))."""
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
