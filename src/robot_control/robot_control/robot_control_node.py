import threading

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String
from std_srvs.srv import Trigger

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import GripperState

from robot_control.doosan_driver import DoosanDriver, DoosanRobotControl
from robot_control.rg2_client import RG2Client, RG2Status
from robot_control.safety_monitor import (
    DoosanRobotState,
    FaultPrefix,
    SafetyMonitor,
    SafetyState,
)
from robot_control.servo_loop import HandApproachServo, ServoLoop
from robot_control.task_executor import TaskExecutor

NAMED_POSE_NAMES = ('home', 'front', 'up', 'down', 'watch', 'handover_safe')


def _declare_double_array(node, name, default):
    node.declare_parameter(
        name, default, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))


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
        self.declare_parameter('servo.z_close_m', 0.01)
        self.declare_parameter('servo.diverge_factor', 1.2)
        self.declare_parameter('servo.diverge_window', 3)
        # TODO(확정 필요): ToolTrack.orientation이 "목표 TCP 자세"를 의미하는지
        # 아직 합의되지 않았다. 확정되기 전에는 절대 orientation을 상대 회전 오차로
        # 오용하지 않도록 yaw 제어 자체를 잠근다(기본 false -> yaw_rate 항상 0).
        self.declare_parameter('servo.enable_yaw_control', False)

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

        self.declare_parameter('safety.external_torque_threshold_nm', 20.0)
        self.declare_parameter('safety.fault_stop_mode', 1)  # DR_QSTOP: Quick stop Cat.2
        self.declare_parameter('safety.state_poll_period_s', 0.1)
        self.declare_parameter('gripper_poll_period_s', 0.5)

        # servo_pick 실제 하드웨어 실행을 위한 별도 게이트. hardware_enabled=true여도
        # 이 값이 false면 servo_pick Goal 자체를 거부한다 (기본값 false).
        # 이유: 현재 ToolTrack.pose는 base_link 절대좌표로 정의되어 있는데
        # (handover_interfaces/msg/ToolTrack.msg), ServoLoop는 이를 TCP(그리퍼) 기준
        # xy 오차로 가정하고 P 제어를 수행한다 (servo_loop.py 상단 주석 참고). 이 좌표
        # 변환이 실제로 구현·검증되기 전까지는 실제 RT 속도 명령을 로봇에 보내면 안 된다.
        self.declare_parameter('servo_pick.hardware_ready', False)
        self.declare_parameter('servo_pick.rt_ip', '192.168.137.100')
        self.declare_parameter('servo_pick.rt_port', 12347)
        self.declare_parameter('servo_pick.rt_control_period_s', 0.01)
        _declare_double_array(
            self, 'servo_pick.speedl_acc', [200.0, 200.0, 200.0, 60.0, 60.0, 60.0])
        # ToolTrack이 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id.
        # TF 변환이 구현되지 않았으므로 다른 frame_id는 거부한다 (_compute_tool_track_tcp_offset).
        self.declare_parameter('servo_pick.tool_track_frame_id', 'base_link')
        # TCP 위치 캐시 샘플의 나이(초)가 이 값보다 크면 오래됐다고 보고 사용하지
        # 않는다 (서비스 왕복 시간이 아니라 _tcp_pose_cache 샘플 자체의 나이를
        # 뜻한다 - _on_tcp_pose_refresh_timer/_get_current_tcp_posx 참고). 하드웨어
        # 캘리브레이션 값이 아니라 RT 루프에 맞는 통신 타이밍 설정이다.
        self.declare_parameter('servo_pick.tcp_pose_max_age_s', 0.2)
        # TCP 위치 캐시를 갱신하는 주기 - ToolTrack 콜백마다 동기 호출하는 대신
        # rate-limited하게 별도 타이머로 갱신한다 (GetCurrentPosx 과부하 방지).
        self.declare_parameter('servo_pick.tcp_pose_refresh_period_s', 0.05)

        # handover_approach: handover_safe 도착 후 /vision/hand_pose(작업자 손 위치)를
        # 향해 servo_pick과 같은 PBVS 패턴으로 접근하다 stop_distance_m 이내가 되면
        # 멈춘다(그리퍼 동작 없음 - 이후 handover_hold가 당김을 기다린다). 사람에게
        # 접근하는 동작이라 servo_pick과 별도 네임스페이스로 두어 더 보수적으로
        # 튜닝할 수 있게 한다.
        # hardware_ready는 servo_pick.hardware_ready와 같은 이유로 기본 false다:
        # hand_pose(vision_node._track_hand)가 아직 미구현(NotImplementedError)이라
        # frame_id/orientation 의미가 검증되지 않았다 - 확정 전까지 실제 RT 속도
        # 명령 발행을 금지한다.
        self.declare_parameter('handover_approach.hardware_ready', False)
        # 사용자가 지정한 접근 정지 거리(5cm) - 실측 협의값.
        self.declare_parameter('handover_approach.stop_distance_m', 0.05)
        self.declare_parameter('handover_approach.v_max', 0.15)
        self.declare_parameter('handover_approach.kp_xy', 1.0)
        self.declare_parameter('handover_approach.timeout_s', 10.0)
        self.declare_parameter('handover_approach.t_lost_s', 0.5)
        self.declare_parameter('handover_approach.diverge_factor', 1.2)
        self.declare_parameter('handover_approach.diverge_window', 3)
        self.declare_parameter('handover_approach.rt_control_period_s', 0.01)
        # hand_pose가 base_link 절대좌표라는 계약을 검증하는 유일한 허용 frame_id.
        # TF 변환이 구현되지 않았으므로 다른 frame_id는 거부한다
        # (_compute_hand_pose_tcp_offset).
        self.declare_parameter('handover_approach.hand_pose_frame_id', 'base_link')

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
            z_close_m=self.get_parameter('servo.z_close_m').value,
            diverge_factor=self.get_parameter('servo.diverge_factor').value,
            diverge_window=self.get_parameter('servo.diverge_window').value,
            enable_yaw_control=bool(self.get_parameter('servo.enable_yaw_control').value),
        )

        self.hand_approach_servo = HandApproachServo(
            kp_xy=self.get_parameter('handover_approach.kp_xy').value,
            v_max=self.get_parameter('handover_approach.v_max').value,
            timeout_s=self.get_parameter('handover_approach.timeout_s').value,
            t_lost_s=self.get_parameter('handover_approach.t_lost_s').value,
            stop_distance_m=self.get_parameter('handover_approach.stop_distance_m').value,
            diverge_factor=self.get_parameter('handover_approach.diverge_factor').value,
            diverge_window=self.get_parameter('handover_approach.diverge_window').value,
        )

        # DoosanDriver 초기화 실패 시 즉시 FAULT를 선언해야 하므로, 발행자를 먼저 만든다.
        self.pub_gripper_state = self.create_publisher(GripperState, '/gripper/state', 10)
        self.pub_fault = self.create_publisher(String, '/robot/fault', 10)

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

        # TCP 위치 캐시 - _on_tcp_pose_refresh_timer가 rate-limited하게 채우고,
        # _get_current_tcp_posx()는 이 캐시만 읽는다 (ToolTrack 콜백에서 매번 동기
        # 서비스 호출을 하지 않기 위함).
        self._tcp_pose_cache = None
        self._tcp_pose_request_in_flight = False
        # servo_pick 또는 handover_approach가 실제로 실행 중일 때만 TCP 위치를
        # 갱신한다 - 불필요한 조회로 executor 스레드를 낭비하거나 안전상태 polling을
        # 지연시키지 않기 위함이다.
        self._tcp_tracking_active = False
        self._gripper_timer = self.create_timer(
            self.get_parameter('gripper_poll_period_s').value,
            self._on_gripper_timer, callback_group=self.sensor_callback_group)
        self._state_poll_timer = self.create_timer(
            self.get_parameter('safety.state_poll_period_s').value,
            self._on_state_poll_timer, callback_group=self.sensor_callback_group)
        self._tcp_pose_timer = self.create_timer(
            self.get_parameter('servo_pick.tcp_pose_refresh_period_s').value,
            self._on_tcp_pose_refresh_timer, callback_group=self.sensor_callback_group)

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
            value = self.get_parameter(f'named_poses.{name}').value
            self._named_poses[name] = list(value) if value else []

    def _init_doosan_driver(self):
        """hardware_enabled=true일 때 DoosanDriver를 생성한다.

        생성에 실패하면(예: dsr_msgs2 미설치) 즉시 safety_state=FAULT를 선언해
        goal_callback이 이후의 모든 Goal을 거부하도록 한다 (하드웨어 경계가 없는
        상태로 조용히 dry_run처럼 동작하지 않는다).
        """
        self._doosan = None
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
        reason = self._check_fault(state)
        if reason is not None and reason != self._last_fault_reason:
            self._declare_fault(reason)

    def _on_gripper_timer(self):
        width_mm, grip_detected = self.rg2_client.get_state()
        msg = GripperState()
        msg.width_mm = width_mm
        msg.grip_detected = grip_detected
        self.pub_gripper_state.publish(msg)

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
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
