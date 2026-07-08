import os
import threading
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

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
from robot_control.servo_loop import ServoLoop
from robot_control.task_executor import TaskExecutor

NAMED_POSE_NAMES = ('home', 'front', 'up', 'down', 'watch', 'handover_safe')


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
        self.declare_parameter('servo.n_stable', 5)
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
        self.declare_parameter('servo.diverge_n', 5)
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

        # base_link -> link_6 TF를 계속 방송한다 - vision_node/tool_detection_node가
        # "이미지가 찍힌 시각의 로봇 자세"를 tf2로 조회할 수 있게 하기 위함이다
        # (로봇이 이동 중이어도 검출 시점의 좌표를 얻을 수 있음). link_6 이후
        # (link_6 -> camera_link 등)는 각 카메라 노드가 자신의 hand-eye 캘리브레이션으로
        # 별도 방송/계산한다 - 여기서는 로봇 자세만 책임진다.
        # 2026-07-08 실기 검증 중 발견: dsr_description2(m0609) URDF는 "flange"라는
        # 프레임을 정의하지 않는다(joint_6/link_6에서 체인이 끝남, xacro 확인) - 이
        # 노드가 원래 발행하던 'flange' 자식 프레임이 실제 로봇 상태 발행과 안 이어져
        # "TF has two or more unconnected trees" 오류가 났었다. 'link_6'으로 정정.
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

        # handover_approach: handover_safe 도착 후 /vision/hand_pose(작업자 손 위치)를
        # 향해 접근하다 stop_distance_m 이내가 되면 멈춘다(그리퍼 동작 없음 - 이후
        # handover_hold가 당김을 기다린다). 실제 접근 로직(movel 기반 단발성 이동)은
        # 아직 구현 전이라 _execute_handover_approach는 게이트 체크 후 TODO를
        # 반환하는 스텁 상태다 - 아래 파라미터는 그 구현이 재사용할 것들이다.
        # hardware_ready는 servo_pick.hardware_ready와 같은 이유로 기본 false다:
        # hand_pose(vision_node._track_hand)가 아직 미구현(NotImplementedError)이라
        # frame_id/orientation 의미가 검증되지 않았다 - 확정 전까지 실제 속도
        # 명령 발행을 금지한다.
        self.declare_parameter('handover_approach.hardware_ready', False)
        # 사용자가 지정한 접근 정지 거리(5cm) - 실측 협의값.
        self.declare_parameter('handover_approach.stop_distance_m', 0.05)
        # hand_pose 수신 대기 타임아웃 - movel 서비스 자체의 왕복 타임아웃(move.timeout_s)과는
        # 별개로, "메시지가 아예 안 온다"를 감지하기 위한 것이다.
        self.declare_parameter('handover_approach.timeout_s', 10.0)
        # hand_pose가 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id.
        # TF 변환이 구현되지 않았으므로 다른 frame_id는 거부한다.
        self.declare_parameter('handover_approach.hand_pose_frame_id', 'base_link')
        # movel 이동 속도 - 사람에게 접근하는 동작이라 실측 전까지 보수적으로 낮게 둔다
        # (named pose 이동인 move.vel_deg_s=30deg/s보다 훨씬 느린 속도). hardware_ready가
        # 기본 false로 게이트되어 있어 실측/검증 전에는 어차피 실제 명령이 나가지 않는다.
        self.declare_parameter('handover_approach.vel_mm_s', 30.0)
        self.declare_parameter('handover_approach.acc_mm_s2', 100.0)

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
            diverge_n=self.get_parameter('servo.diverge_n').value,
            cov_threshold=self.get_parameter('servo.cov_threshold').value,
            q_pos=self.get_parameter('servo.kalman_q_pos').value,
            q_vel=self.get_parameter('servo.kalman_q_vel').value,
            r_xy=self.get_parameter('servo.kalman_r_xy').value,
            r_z=self.get_parameter('servo.kalman_r_z').value,
            p0_vel_reset=self.get_parameter('servo.kalman_p0_vel_reset').value,
        )

        # DoosanDriver 초기화 실패 시 즉시 FAULT를 선언해야 하므로, 발행자를 먼저 만든다.
        self.pub_gripper_state = self.create_publisher(GripperState, '/gripper/state', 10)
        self.pub_fault = self.create_publisher(String, '/robot/fault', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

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

        # TCP 위치 캐시 - _on_tf_broadcast_timer가 TF 방송용으로 조회한 GetCurrentPosx
        # 결과를 함께 채운다(아래 타이머 설명 참고). _get_current_tcp_posx()는 이
        # 캐시만 읽는다 (ToolTrack 콜백에서 매번 동기 서비스 호출을 하지 않기 위함).
        self._tcp_pose_cache = None
        # _on_tf_broadcast_timer 전체(TF 방송 + TCP 캐시 갱신)를 감싸는 단일
        # in-flight 가드 - 두 용도가 같은 GetCurrentPosx 서비스 호출 하나를
        # 공유하므로 플래그도 하나만 있으면 된다(과거에는 두 타이머가 각자 별도
        # 요청을 겹쳐 보내 스레드 고갈을 유발했다).
        self._tcp_pose_request_in_flight = False
        # handover_approach가 /vision/hand_pose 1개를 받을 때까지 임시로 채워두는
        # 슬롯 (_on_hand_pose_received/_wait_for_hand_pose 참고).
        self._pending_hand_pose = None
        # servo_pick 또는 handover_approach가 실제로 실행 중일 때만 TCP 위치를
        # 캐시에 반영한다 - 불필요한 상태 갱신을 피하기 위함이다(TF 방송 자체는
        # 이 값과 무관하게 항상 진행된다).
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
        self.safety_monitor.declare_fault(reason)

    def destroy_node(self):
        if getattr(self, '_drfl_force_monitor', None) is not None:
            self._drfl_force_monitor.stop()
        super().destroy_node()

    # ---- goal 수락/취소 ----

    def _goal_callback(self, goal_request):
        if self.safety_state != SafetyState.NORMAL:
            self.get_logger().warn(f'Goal 거부 - safety_state={self.safety_state}')
            return GoalResponse.REJECT
        if goal_request.task_type not in self._handlers:
            self.get_logger().warn(f'Goal 거부 - 알 수 없는 task_type: {goal_request.task_type}')
            return GoalResponse.REJECT
        if (goal_request.task_type == 'servo_pick' and self.hardware_enabled
                and not self.get_parameter('servo_pick.hardware_ready').value):
            self.get_logger().warn(
                'Goal 거부 - servo_pick.hardware_ready=false (ToolTrack 좌표 변환 미검증)')
            return GoalResponse.REJECT
        if (goal_request.task_type == 'handover_approach' and self.hardware_enabled
                and not self.get_parameter('handover_approach.hardware_ready').value):
            self.get_logger().warn(
                'Goal 거부 - handover_approach.hardware_ready=false (hand_pose 좌표 변환 미검증)')
            return GoalResponse.REJECT
        # goal 수락 경쟁(TOCTOU) 방지: 락 안에서 원자적으로 하나만 예약한다.
        with self._goal_lock:
            if self._goal_reserved:
                self.get_logger().warn('Goal 거부 - 이미 실행 중(또는 취소 처리 중)인 goal이 있습니다.')
                return GoalResponse.REJECT
            self._goal_reserved = True
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
        reason = self._check_fault(state)
        if reason is not None and reason != self._last_fault_reason:
            self._declare_fault(reason)

    def _on_gripper_timer(self):
        width_mm, grip_detected = self.rg2_client.get_state()
        msg = GripperState()
        msg.width_mm = width_mm
        msg.grip_detected = grip_detected
        self.pub_gripper_state.publish(msg)

    def _on_tf_broadcast_timer(self):
        """base_link -> link_6(tf_broadcast.child_frame_id) TF를 방송하고, 같은
        GetCurrentPosx 조회 결과로 servo_pick의 TCP 위치 캐시(_tcp_pose_cache)도
        함께 채운다. TF는 로봇이 이동 중에도 vision_node/tool_detection_node가
        "이미지가 찍힌 시각"의 로봇 자세를 tf2로 조회할 수 있게 하기 위함이고(그 뒤
        link_6 -> camera_link는 각 카메라 노드의 hand-eye 캘리브레이션이 별도로
        책임진다), TCP 캐시는 칼만 ServoLoop.step()이 매 RT tick 읽는 값이다. dry-run
        (hardware_enabled=false)에서는 실제 자세가 없으므로 아무것도 하지 않는다.

        2026-07-08 실기 검증 중 발견: 원래 이 타이머(50Hz, TF 전용)와
        _on_tcp_pose_refresh_timer(20Hz, TCP 캐시 전용)가 각자 in-flight 가드를
        갖고 독립적으로 GetCurrentPosx를 호출했다. servo_pick이 시작해 두 타이머가
        동시에 뛰기 시작하면 응답이 느려지는 순간(doosan_driver._wait_for_future는
        최대 future_wait_timeout_s=2초까지 스레드를 블로킹) 겹친 호출들이
        MultiThreadedExecutor(num_threads=4)의 스레드를 전부 점유해버려 TF 방송이
        통째로 멈췄다(실기에서 5초+ 정지 확인 - servo_pick 진입과 동시에 발생,
        결과적으로 vision_node의 TF lookup도 "extrapolation into the future"로
        계속 실패해 ToolTrack이 아예 발행되지 못했다). 두 폴링을 이 타이머 하나로
        합쳐 GetCurrentPosx 호출 자체를 하나의 스트림으로 만들었다 - in-flight
        가드는 여전히 필요하다(호출 자체가 timer period보다 오래 걸릴 수 있으므로)."""
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
        from scipy.spatial.transform import Rotation  # dry-run에서는 필요 없는 지연 임포트
        x_mm, y_mm, z_mm, a_deg, b_deg, c_deg = pos6
        quat = Rotation.from_euler('ZYZ', [a_deg, b_deg, c_deg], degrees=True).as_quat()

        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('tf_broadcast.parent_frame_id').value
        msg.child_frame_id = self.get_parameter('tf_broadcast.child_frame_id').value
        msg.transform.translation.x = x_mm / 1000.0
        msg.transform.translation.y = y_mm / 1000.0
        msg.transform.translation.z = z_mm / 1000.0
        msg.transform.rotation.x = float(quat[0])
        msg.transform.rotation.y = float(quat[1])
        msg.transform.rotation.z = float(quat[2])
        msg.transform.rotation.w = float(quat[3])
        self._tf_broadcaster.sendTransform(msg)

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
