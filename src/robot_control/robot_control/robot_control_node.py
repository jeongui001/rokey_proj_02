import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String
from std_srvs.srv import Trigger

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import GripperState, ToolTrack

from robot_control.servo_loop import ServoLoop

NAMED_POSE_NAMES = ('home', 'front', 'up', 'down', 'watch', 'handover_safe', 'place_down')


class SafetyState:
    NORMAL = 'NORMAL'
    PROTECTIVE_STOP = 'PROTECTIVE_STOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'
    FAULT = 'FAULT'


# 안전상태 우선순위: 숫자가 클수록 더 심각하다. task_manager.task_manager_node.Safety와
# 합의된 순서(NORMAL < RECOVERY_REQUIRED < PROTECTIVE_STOP < FAULT < EMERGENCY_STOP)와
# 일치시킨다 (robot_control에는 RECOVERY_REQUIRED가 없다).
SAFETY_STATE_PRIORITY = {
    SafetyState.NORMAL: 0,
    SafetyState.PROTECTIVE_STOP: 1,
    SafetyState.FAULT: 2,
    SafetyState.EMERGENCY_STOP: 3,
}


class FaultPrefix:
    """task_manager와의 계약: /robot/fault 문자열 접두어 (task_manager_node._is_protective_stop 등과 합의됨)."""
    PROTECTIVE_STOP = 'PROTECTIVE_STOP: '
    EMERGENCY_STOP = 'EMERGENCY_STOP: '
    FAULT = 'FAULT: '


class DoosanRobotState:
    """dsr_msgs2/srv/system/GetRobotState.srv 응답 enum (실제 소스에서 확인, 추측 아님)."""
    INITIALIZING = 0
    STANDBY = 1
    MOVING = 2
    SAFE_OFF = 3
    TEACHING = 4
    SAFE_STOP = 5
    EMERGENCY_STOP = 6
    HOMMING = 7
    RECOVERY = 8
    SAFE_STOP2 = 9
    SAFE_OFF2 = 10
    NOT_READY = 15


class DoosanRobotControl:
    """dsr_msgs2/srv/system/SetRobotControl.srv 요청 enum (실제 소스에서 확인)."""
    CONTROL_INIT_CONFIG = 0
    CONTROL_ENABLE_OPERATION = 1
    CONTROL_RESET_SAFET_STOP = 2
    CONTROL_RESET_SAFET_OFF = 3
    CONTROL_RECOVERY_SAFE_STOP = 4
    CONTROL_RECOVERY_SAFE_OFF = 5
    CONTROL_RECOVERY_BACKDRIVE = 6
    CONTROL_RESET_RECOVERY = 7


class RG2Client:
    """OnRobot RG2/RG6 그리퍼를 Modbus TCP(Tool Changer)로 제어한다.

    레지스터 프로토콜은 DoosanBootcamp의
    ``dsr_rokey/pick_and_place_voice/robot_control/onrobot.py`` (RG 클래스)에서 확인한
    실제 프로토콜을 그대로 따른다 (추측 없음):
      - write_registers(address=0, values=[force(1/10 N), width(1/10 mm), command], unit=65)
        command: 1=grip, 8=stop, 16=grip_w_offset (onrobot.py는 open/close/move 모두 16 사용)
      - read_holding_registers(address=267, count=1, unit=65) -> width(1/10 mm)
      - read_holding_registers(address=268, count=1, unit=65) -> status bit0=busy, bit1=grip_detected

    hardware_enabled=False(기본값)에서는 실제 Modbus 통신을 절대 하지 않고, 안전한
    시뮬레이션 값만 상태로 유지/반환한다.
    """

    _CMD_GRIP_W_OFFSET = 16
    MAX_WIDTH_MM = {'rg2': 110.0, 'rg6': 160.0}
    MAX_FORCE_N = {'rg2': 40.0, 'rg6': 120.0}

    def __init__(self, ip: str, port: int = 502, hardware_enabled: bool = False, gripper: str = 'rg2'):
        self.ip = ip
        self.port = port
        self.hardware_enabled = hardware_enabled
        self.gripper = gripper
        self._client = None
        self._sim_width_mm = self.MAX_WIDTH_MM.get(gripper, 110.0)
        self._sim_grip_detected = False

    def _ensure_connected(self):
        if self._client is not None:
            return self._client
        from pymodbus.client.sync import ModbusTcpClient  # 지연 임포트 - dry_run에서는 불필요
        self._client = ModbusTcpClient(
            self.ip, port=self.port, stopbits=1, bytesize=8, parity='E',
            baudrate=115200, timeout=1)
        self._client.connect()
        return self._client

    def open(self) -> None:
        if not self.hardware_enabled:
            self._sim_width_mm = self.MAX_WIDTH_MM.get(self.gripper, 110.0)
            self._sim_grip_detected = False
            return
        client = self._ensure_connected()
        max_width = int(self.MAX_WIDTH_MM.get(self.gripper, 110.0) * 10)
        max_force = int(self.MAX_FORCE_N.get(self.gripper, 40.0) * 10)
        client.write_registers(
            address=0, values=[max_force, max_width, self._CMD_GRIP_W_OFFSET], unit=65)

    def close(self, width_mm: float, force_n: float) -> None:
        if not self.hardware_enabled:
            self._sim_width_mm = width_mm
            self._sim_grip_detected = True
            return
        client = self._ensure_connected()
        client.write_registers(
            address=0,
            values=[int(force_n * 10), int(width_mm * 10), self._CMD_GRIP_W_OFFSET],
            unit=65)

    def get_state(self):
        """(width_mm: float, grip_detected: bool) 튜플을 반환한다."""
        if not self.hardware_enabled:
            return (self._sim_width_mm, self._sim_grip_detected)
        client = self._ensure_connected()
        width_result = client.read_holding_registers(address=267, count=1, unit=65)
        status_result = client.read_holding_registers(address=268, count=1, unit=65)
        width_mm = width_result.registers[0] / 10.0
        status_bits = format(status_result.registers[0], '016b')
        grip_detected = bool(int(status_bits[-2]))
        return (width_mm, grip_detected)


class DoosanDriver:
    """dsr_msgs2 서비스/토픽 호출 경계. hardware_enabled=True일 때만 생성/사용된다.

    실제 서비스·메시지 이름과 필드는 DoosanBootcamp(~/ros_ws/src/DoosanBootcamp)와
    doosan-robot2(~/cobot_ws/src/doosan-robot2)의 dsr_msgs2 .srv/.msg 정의를 직접 읽어
    확인했다 (추측 없음). 노드가 launch에서 namespace=<robot_id>로 기동된다는 전제로
    상대 경로 서비스/토픽 이름을 사용한다 (DSR_ROBOT2.py의 _srv_name_prefix='' 관례와 동일).
    """

    def __init__(self, node):
        try:
            from dsr_msgs2.srv import (
                MoveJoint, MoveLine, MoveStop, GetRobotState, SetRobotControl,
                GetExternalTorque, GetToolForce, ConnectRtControl, StartRtControl,
                StopRtControl, DisconnectRtControl, TaskComplianceCtrl, ReleaseComplianceCtrl,
            )
            from dsr_msgs2.msg import SpeedlRtStream
        except ImportError as exc:
            raise RuntimeError(
                'dsr_msgs2를 임포트할 수 없습니다. hardware_enabled=true로 실행하려면 '
                'Doosan ROS2 드라이버(dsr_msgs2)가 설치된 워크스페이스를 source 해야 합니다.'
            ) from exc

        self._node = node
        self._MoveJoint = MoveJoint
        self._MoveLine = MoveLine
        self._MoveStop = MoveStop
        self._GetRobotState = GetRobotState
        self._SetRobotControl = SetRobotControl
        self._GetExternalTorque = GetExternalTorque
        self._GetToolForce = GetToolForce
        self._ConnectRtControl = ConnectRtControl
        self._StartRtControl = StartRtControl
        self._StopRtControl = StopRtControl
        self._DisconnectRtControl = DisconnectRtControl
        self._TaskComplianceCtrl = TaskComplianceCtrl
        self._ReleaseComplianceCtrl = ReleaseComplianceCtrl
        self._SpeedlRtStream = SpeedlRtStream

        # 주의: robot_control 노드 자신은 task_manager와 이미 합의된 절대 경로 인터페이스
        # (/robot_task, /robot/fault, /gripper/state, /robot/recover)를 노출해야 하므로,
        # 노드 전체를 namespace=<robot_id>로 띄울 수 없다 (그러면 이 인터페이스들도 함께
        # 네임스페이스가 붙어버려 계약이 깨진다). 따라서 dsr_msgs2 쪽 이름만 robot_id를
        # 접두어로 붙인 절대 경로로 구성한다 (부트캠프의 상대 경로+node namespace 관례 대신).
        robot_id = node.get_parameter('robot_id').value
        prefix = f'/{robot_id}'
        cb_group = node.hardware_callback_group
        self._cli_move_joint = node.create_client(
            MoveJoint, f'{prefix}/motion/move_joint', callback_group=cb_group)
        self._cli_move_line = node.create_client(
            MoveLine, f'{prefix}/motion/move_line', callback_group=cb_group)
        self._cli_move_stop = node.create_client(
            MoveStop, f'{prefix}/motion/move_stop', callback_group=cb_group)
        self._cli_get_robot_state = node.create_client(
            GetRobotState, f'{prefix}/system/get_robot_state', callback_group=cb_group)
        self._cli_set_robot_control = node.create_client(
            SetRobotControl, f'{prefix}/system/set_robot_control', callback_group=cb_group)
        self._cli_get_ext_torque = node.create_client(
            GetExternalTorque, f'{prefix}/aux_control/get_external_torque', callback_group=cb_group)
        self._cli_get_tool_force = node.create_client(
            GetToolForce, f'{prefix}/aux_control/get_tool_force', callback_group=cb_group)
        self._cli_connect_rt = node.create_client(
            ConnectRtControl, f'{prefix}/realtime/connect_rt_control', callback_group=cb_group)
        self._cli_start_rt = node.create_client(
            StartRtControl, f'{prefix}/realtime/start_rt_control', callback_group=cb_group)
        self._cli_stop_rt = node.create_client(
            StopRtControl, f'{prefix}/realtime/stop_rt_control', callback_group=cb_group)
        self._cli_disconnect_rt = node.create_client(
            DisconnectRtControl, f'{prefix}/realtime/disconnect_rt_control', callback_group=cb_group)
        self._cli_task_compliance = node.create_client(
            TaskComplianceCtrl, f'{prefix}/force/task_compliance_ctrl', callback_group=cb_group)
        self._cli_release_compliance = node.create_client(
            ReleaseComplianceCtrl, f'{prefix}/force/release_compliance_ctrl', callback_group=cb_group)
        self._pub_speedl_rt = node.create_publisher(SpeedlRtStream, f'{prefix}/speedl_rt_stream', 10)

    def _wait_for_future(self, future, timeout_s):
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            return None
        return future.result()

    def _call_move_with_cancel(self, client, request, goal_handle, poll_interval_s, timeout_s):
        if not client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().error(f'{client.srv_name} 서비스에 연결할 수 없습니다.')
            return False
        future = client.call_async(request)
        start = time.monotonic()
        while rclpy.ok():
            if future.done():
                break
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self._node.get_logger().warn('이동 취소 요청 감지 - move_stop 호출')
                self.stop(self._node.get_parameter('safety.fault_stop_mode').value)
                # move_stop 이후에도 원래 future가 짧게 더 완료되길 기다린다 (무한 대기 방지).
                self._wait_for_future(future, 2.0)
                return False
            if self._node.safety_state != SafetyState.NORMAL:
                # cancel뿐 아니라 Fault도 주기적으로 확인한다 - 이동 완료 응답이 성공으로
                # 오는 순간과 Fault 발생이 겹치는 경합을 줄인다.
                self._node.get_logger().warn('이동 대기 중 안전상태 비정상 감지 - move_stop 호출')
                self.stop(self._node.get_parameter('safety.fault_stop_mode').value)
                self._wait_for_future(future, 2.0)
                return False
            if time.monotonic() - start > timeout_s:
                self._node.get_logger().error('move 서비스 응답 타임아웃 - move_stop 호출')
                self.stop(self._node.get_parameter('safety.fault_stop_mode').value)
                return False
            time.sleep(poll_interval_s)
        if not future.done():
            return False
        if self._node.safety_state != SafetyState.NORMAL:
            return False
        response = future.result()
        return bool(response is not None and response.success)

    def move_joint(self, goal_handle, pos_deg6, vel, acc, radius_mm=0.0,
                   sync_type=0, poll_interval_s=0.05, timeout_s=30.0) -> bool:
        request = self._MoveJoint.Request()
        request.pos = [float(v) for v in pos_deg6]
        request.vel = float(vel)
        request.acc = float(acc)
        request.time = 0.0
        request.radius = float(radius_mm)
        request.mode = 0  # MOVE_MODE_ABSOLUTE
        request.blend_type = 0  # BLENDING_SPEED_TYPE_DUPLICATE
        request.sync_type = int(sync_type)
        return self._call_move_with_cancel(
            self._cli_move_joint, request, goal_handle, poll_interval_s, timeout_s)

    def move_line(self, goal_handle, pos6, vel2, acc2, ref=0, radius_mm=0.0,
                  sync_type=0, poll_interval_s=0.05, timeout_s=30.0) -> bool:
        request = self._MoveLine.Request()
        request.pos = [float(v) for v in pos6]
        request.vel = [float(vel2[0]), float(vel2[1])]
        request.acc = [float(acc2[0]), float(acc2[1])]
        request.time = 0.0
        request.radius = float(radius_mm)
        request.ref = int(ref)
        request.mode = 0  # DR_MV_MOD_ABS
        request.blend_type = 0
        request.sync_type = int(sync_type)
        return self._call_move_with_cancel(
            self._cli_move_line, request, goal_handle, poll_interval_s, timeout_s)

    def stop(self, stop_mode=1) -> bool:
        if not self._cli_move_stop.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().error('motion/move_stop 서비스에 연결할 수 없습니다.')
            return False
        request = self._MoveStop.Request()
        request.stop_mode = int(stop_mode)
        response = self._wait_for_future(self._cli_move_stop.call_async(request), 2.0)
        return bool(response is not None and response.success)

    def get_robot_state(self):
        if not self._cli_get_robot_state.wait_for_service(timeout_sec=1.0):
            return None
        response = self._wait_for_future(
            self._cli_get_robot_state.call_async(self._GetRobotState.Request()), 2.0)
        if response is None or not response.success:
            return None
        return int(response.robot_state)

    def get_external_torque(self):
        if not self._cli_get_ext_torque.wait_for_service(timeout_sec=1.0):
            return None
        response = self._wait_for_future(
            self._cli_get_ext_torque.call_async(self._GetExternalTorque.Request()), 2.0)
        if response is None or not response.success:
            return None
        return list(response.ext_torque)

    def get_tool_force(self, ref=0):
        # GetToolForce.srv는 DR_BASE(0)/DR_TOOL(1)/DR_WORLD(2)를 나열하지만, 이 노드에서는
        # DR_TOOL(1)을 허용하지 않는다 (호출측 요구사항: BASE/WORLD만 사용). 잘못된 값은
        # 서비스 호출 전에 거부한다.
        if int(ref) not in (0, 2):
            self._node.get_logger().error(
                f'get_tool_force: 지원하지 않는 ref={ref} (BASE=0 또는 WORLD=2만 허용)')
            return None
        if not self._cli_get_tool_force.wait_for_service(timeout_sec=1.0):
            return None
        request = self._GetToolForce.Request()
        request.ref = int(ref)
        response = self._wait_for_future(self._cli_get_tool_force.call_async(request), 2.0)
        if response is None or not response.success:
            return None
        return list(response.tool_force)

    def set_robot_control(self, code) -> bool:
        if not self._cli_set_robot_control.wait_for_service(timeout_sec=1.0):
            return False
        request = self._SetRobotControl.Request()
        request.robot_control = int(code)
        response = self._wait_for_future(self._cli_set_robot_control.call_async(request), 2.0)
        return bool(response is not None and response.success)

    def open_rt_session(self) -> bool:
        """RT 세션을 연다. StartRtControl의 success가 확인된 경우에만 True를 반환한다.

        ConnectRtControl 실패는 즉시 예외로 처리하지 않는다: DSR_ROBOT2.connect_rt_control()의
        코드 주석은 "RT 연결은 dsr_hw_interface2가 이미 맺어뒀을 수 있으니 재호출에 주의하라"고
        경고하므로, 이미 연결되어 있어 재연결 요청이 실패/거부되는 경우가 있을 수 있다
        (TODO(확인 필요): 실제 M0609 브링업에서 확인 필요). 대신 StartRtControl의 응답을
        최종 판단 기준으로 삼는다 - 이것이 실패하면 예외를 던져 servo_pick을 중단시킨다.
        """
        ip = self._node.get_parameter('servo_pick.rt_ip').value
        port = self._node.get_parameter('servo_pick.rt_port').value
        if self._cli_connect_rt.wait_for_service(timeout_sec=1.0):
            request = self._ConnectRtControl.Request()
            request.ip_address = ip
            request.port = int(port)
            response = self._wait_for_future(self._cli_connect_rt.call_async(request), 3.0)
            if response is None or not response.success:
                self._node.get_logger().warn(
                    'connect_rt_control 실패 또는 응답 없음 - dsr_hw_interface2가 이미 '
                    '연결되어 있을 수 있으므로 계속 진행하고 start_rt_control 결과로 최종 판단한다.')
        else:
            self._node.get_logger().warn('realtime/connect_rt_control 서비스를 찾을 수 없습니다.')

        if not self._cli_start_rt.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('realtime/start_rt_control 서비스에 연결할 수 없습니다.')
        response = self._wait_for_future(
            self._cli_start_rt.call_async(self._StartRtControl.Request()), 3.0)
        if response is None or not response.success:
            raise RuntimeError('start_rt_control이 실패했습니다 (RT 세션이 시작되지 않음).')
        return True

    def close_rt_session(self):
        if self._cli_stop_rt.wait_for_service(timeout_sec=1.0):
            self._wait_for_future(self._cli_stop_rt.call_async(self._StopRtControl.Request()), 3.0)
        if self._cli_disconnect_rt.wait_for_service(timeout_sec=1.0):
            self._wait_for_future(
                self._cli_disconnect_rt.call_async(self._DisconnectRtControl.Request()), 3.0)

    def publish_speedl_rt(self, cmd):
        # TODO(단위 미확인 - 추측 변환 금지): dsr_msgs2/msg/SpeedlRtStream.msg에는 단위
        # 주석이 없고(MoveLine.srv는 "[mm/sec],[deg/sec]"를 명시하지만 SpeedlRtStream은
        # 없음), DSR_ROBOT2.py의 speedl_rt() 래퍼도 단위를 문서화하지 않는다.
        # dsr_controller2.cpp의 speedl_rt_cb는 msg.vel/msg.acc를 그대로
        # Drfl->speedl_rt()(비공개 DRFL 바이너리)에 전달할 뿐이라 이 저장소 소스만으로는
        # 실제 단위(mm/s vs m/s 등)를 확정할 수 없었다. 따라서 여기서 값을 임의로
        # 스케일 변환하지 않고 ServoCommand(m/s 기준)를 그대로 넣는다 - 이 값이 실제
        # 단위와 다를 수 있으므로 servo_pick.hardware_ready=false로 발행 자체를 막는다
        # (robot_control_node._execute_servo_pick 참고). 실기 검증 후에만 활성화할 것.
        msg = self._SpeedlRtStream()
        msg.vel = [cmd.vx, cmd.vy, cmd.vz, 0.0, 0.0, cmd.yaw_rate]
        acc = self._node.get_parameter('servo_pick.speedl_acc').value
        msg.acc = list(acc)
        msg.time = self._node.get_parameter('servo_pick.rt_control_period_s').value
        self._pub_speedl_rt.publish(msg)

    def enable_compliance(self):
        if not self._cli_task_compliance.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('force/task_compliance_ctrl 서비스에 연결할 수 없습니다.')
        request = self._TaskComplianceCtrl.Request()
        request.stx = list(self._node.get_parameter('handover_hold.compliance_stiffness').value)
        request.ref = 0  # DR_BASE
        request.time = self._node.get_parameter('handover_hold.compliance_transition_s').value
        response = self._wait_for_future(self._cli_task_compliance.call_async(request), 3.0)
        if response is None or not response.success:
            raise RuntimeError('task_compliance_ctrl 호출이 실패했습니다.')

    def disable_compliance(self):
        if not self._cli_release_compliance.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('force/release_compliance_ctrl 서비스에 연결할 수 없습니다.')
        self._wait_for_future(
            self._cli_release_compliance.call_async(self._ReleaseComplianceCtrl.Request()), 3.0)


def _declare_double_array(node, name, default):
    node.declare_parameter(
        name, default, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))


class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')

        self.declare_parameter('hardware_enabled', False)
        self.declare_parameter('robot_id', 'dsr01')
        self.declare_parameter('rg2_ip', '192.168.1.1')
        self.declare_parameter('rg2_port', 502)
        self.declare_parameter('rg2_gripper', 'rg2')

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

        self.declare_parameter('move.vel_deg_s', 30.0)
        self.declare_parameter('move.acc_deg_s2', 30.0)
        self.declare_parameter('move.line_vel_mm_s', 100.0)
        self.declare_parameter('move.line_vel_deg_s', 30.0)
        self.declare_parameter('move.line_acc_mm_s2', 100.0)
        self.declare_parameter('move.line_acc_deg_s2', 30.0)
        self.declare_parameter('move.ref', 0)
        self.declare_parameter('move.blend_radius_mm', 0.0)
        self.declare_parameter('move.sync_type', 0)
        self.declare_parameter('move.dry_run_duration_s', 0.0)
        self.declare_parameter('move.poll_interval_s', 0.05)
        self.declare_parameter('move.timeout_s', 30.0)

        for name in NAMED_POSE_NAMES:
            _declare_double_array(self, f'named_poses.{name}', [])

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

        self.hardware_enabled = bool(self.get_parameter('hardware_enabled').value)
        self.safety_state = SafetyState.NORMAL
        # 동일 Fault 메시지를 매 폴링 주기(0.1s)마다 반복 발행하지 않기 위한 dedup 키.
        self._last_fault_reason = None
        # robot_state 샘플 식별자. _read_robot_state가 실제로 새 샘플을 얻을 때마다
        # 증가시켜, handover_hold가 같은 샘플을 여러 번 읽어 중복 평가하지 않도록 한다.
        self._robot_state_seq = 0
        self._named_poses = {name: [] for name in NAMED_POSE_NAMES}
        self._refresh_named_poses()

        self.action_callback_group = MutuallyExclusiveCallbackGroup()
        self.sensor_callback_group = ReentrantCallbackGroup()
        self.hardware_callback_group = ReentrantCallbackGroup()

        self.rg2_client = RG2Client(
            ip=self.get_parameter('rg2_ip').value,
            port=self.get_parameter('rg2_port').value,
            hardware_enabled=self.hardware_enabled,
            gripper=self.get_parameter('rg2_gripper').value)

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
        )

        # DoosanDriver 초기화 실패 시 즉시 FAULT를 선언해야 하므로, 발행자를 먼저 만든다.
        self.pub_gripper_state = self.create_publisher(GripperState, '/gripper/state', 10)
        self.pub_fault = self.create_publisher(String, '/robot/fault', 10)

        self._init_doosan_driver()

        # goal 수락 경쟁(TOCTOU) 방지용: goal_callback 안에서 락을 잡고 원자적으로
        # 하나의 goal만 예약한다. execute_callback 종료 시(finally) 예약을 해제한다.
        self._goal_lock = threading.Lock()
        self._goal_reserved = False
        self._active_goal_handle = None
        self._handlers = {
            'move_named': self._execute_move_named,
            'move_pose': self._execute_move_pose,
            'place_down': self._execute_move_named,
            'release_and_retry': self._execute_release_and_retry,
            'servo_pick': self._execute_servo_pick,
            'handover_hold': self._execute_handover_hold,
        }

        self._action_server = ActionServer(
            self, RobotTask, 'robot_task',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.action_callback_group)

        self._latest_robot_state = None
        self._gripper_timer = self.create_timer(
            self.get_parameter('gripper_poll_period_s').value,
            self._on_gripper_timer, callback_group=self.sensor_callback_group)
        self._state_poll_timer = self.create_timer(
            self.get_parameter('safety.state_poll_period_s').value,
            self._on_state_poll_timer, callback_group=self.sensor_callback_group)

        self.recover_srv = self.create_service(
            Trigger, '/robot/recover', self._on_recover, callback_group=self.sensor_callback_group)

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

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

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
        # goal 수락 경쟁(TOCTOU) 방지: 락 안에서 원자적으로 하나만 예약한다.
        with self._goal_lock:
            if self._goal_reserved:
                self.get_logger().warn('Goal 거부 - 이미 실행 중(또는 취소 처리 중)인 goal이 있습니다.')
                return GoalResponse.REJECT
            self._goal_reserved = True
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    # ---- move / place_down / release_and_retry ----

    def _pose_stamped_to_posx(self, pose_stamped):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError as exc:
            self.get_logger().error(f'scipy가 필요합니다 (move_pose 자세 변환): {exc}')
            return None
        position = pose_stamped.pose.position
        orientation = pose_stamped.pose.orientation
        try:
            euler_zyz_deg = Rotation.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ).as_euler('zyz', degrees=True)
        except Exception as exc:  # 통신 오류/예외 시 성공을 반환하지 않는다
            self.get_logger().error(f'move_pose 자세 변환 실패: {exc}')
            return None
        return [position.x * 1000.0, position.y * 1000.0, position.z * 1000.0,
                float(euler_zyz_deg[0]), float(euler_zyz_deg[1]), float(euler_zyz_deg[2])]

    # ---- 하드웨어 정리 경계 (MoveStop / RT 종료 / compliance 해제 / subscription 제거) ----
    #
    # 이 아래 _cleanup_* 함수들은 _safe_call()과 다른 목적을 가진다: _safe_call은
    # 알고리즘 스텁(NotImplementedError)만 걸러내기 위한 것이라 그 용도를 넓히지 않는다.
    # 반면 여기 cleanup 경계는 실제 하드웨어/서비스 호출이라 통신 오류 등 임의의
    # Exception이 발생할 수 있고, 그 예외가 Action 실행 함수 밖으로 새어나가 최종
    # result가 반환되지 못하는 사고(취소/Fault 처리 도중 두 번째 예외가 무방비로
    # 전파되는 경우)를 막아야 한다. 그래서 각 함수는 예외를 명시적으로 잡아 로그를
    # 남기고 성공 여부를 bool로만 반환한다. 여러 번 호출돼도 안전(idempotent)하다 -
    # 이미 정지/종료된 하드웨어에 다시 정지/종료를 요청하는 것은 그 자체로 무해하다.

    def _cleanup_stop_motion(self) -> bool:
        """M0609 MoveStop 정지를 시도한다. 실패/예외 시에도 절대 상위로 전파하지 않는다."""
        if not self.hardware_enabled or self._doosan is None:
            return True
        stop_mode = self.get_parameter('safety.fault_stop_mode').value
        try:
            return bool(self._doosan.stop(stop_mode))
        except Exception as exc:
            self.get_logger().error(f'MoveStop cleanup 중 예외: {exc}')
            return False

    def _cleanup_close_rt_session(self) -> bool:
        """RT 세션 stop/disconnect (_close_rt_session 경계). 실패/예외 시에도 절대
        상위로 전파하지 않는다."""
        try:
            self._close_rt_session()
            return True
        except Exception as exc:
            self.get_logger().error(f'RT 세션 종료 cleanup 중 예외: {exc}')
            return False

    def _cleanup_disable_compliance(self) -> bool:
        """compliance 해제 (_disable_compliance 경계). 실패/예외 시에도 절대 상위로
        전파하지 않는다."""
        try:
            self._disable_compliance()
            return True
        except Exception as exc:
            self.get_logger().error(f'compliance 해제 cleanup 중 예외: {exc}')
            return False

    def _cleanup_destroy_subscription(self, sub) -> bool:
        """servo_pick 중 임시로 만든 ToolTrack subscription을 제거한다."""
        if sub is None:
            return True
        try:
            self.destroy_subscription(sub)
            return True
        except Exception as exc:
            self.get_logger().error(f'subscription 제거 cleanup 중 예외: {exc}')
            return False

    def _dry_run_move(self, goal_handle) -> bool:
        duration_s = self.get_parameter('move.dry_run_duration_s').value
        poll_interval_s = max(self.get_parameter('move.poll_interval_s').value, 0.001)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return False
            if self.safety_state != SafetyState.NORMAL:
                return False
            time.sleep(min(poll_interval_s, max(deadline - time.monotonic(), 0.0)))
        return True

    def _move_joint(self, goal_handle, pos_deg6, vel, acc) -> bool:
        if not self.hardware_enabled:
            return self._dry_run_move(goal_handle)
        if self._doosan is None:
            self.get_logger().error('DoosanDriver가 초기화되지 않았습니다 - move_joint 실패')
            return False
        return self._doosan.move_joint(
            goal_handle, pos_deg6, vel, acc,
            radius_mm=self.get_parameter('move.blend_radius_mm').value,
            sync_type=self.get_parameter('move.sync_type').value,
            poll_interval_s=self.get_parameter('move.poll_interval_s').value,
            timeout_s=self.get_parameter('move.timeout_s').value)

    def _move_line(self, goal_handle, pos6, vel2, acc2) -> bool:
        if not self.hardware_enabled:
            return self._dry_run_move(goal_handle)
        if self._doosan is None:
            self.get_logger().error('DoosanDriver가 초기화되지 않았습니다 - move_line 실패')
            return False
        return self._doosan.move_line(
            goal_handle, pos6, vel2, acc2,
            ref=self.get_parameter('move.ref').value,
            radius_mm=self.get_parameter('move.blend_radius_mm').value,
            sync_type=self.get_parameter('move.sync_type').value,
            poll_interval_s=self.get_parameter('move.poll_interval_s').value,
            timeout_s=self.get_parameter('move.timeout_s').value)

    def _call_move_service(self, goal_handle=None, named_target='', target_pose=None) -> bool:
        """고정 자세(movej) 또는 목표 pose(movel) 이동을 수행한다."""
        if named_target:
            pos = self._named_poses.get(named_target)
            if not pos:
                self.get_logger().error(
                    f"named pose '{named_target}'의 관절값이 설정되지 않았습니다 "
                    f"(파라미터 named_poses.{named_target}). 이동을 수행하지 않습니다.")
                return False
            vel = self.get_parameter('move.vel_deg_s').value
            acc = self.get_parameter('move.acc_deg_s2').value
            success = self._move_joint(goal_handle, pos, vel, acc)
        elif target_pose is not None:
            pos6 = self._pose_stamped_to_posx(target_pose)
            if pos6 is None:
                return False
            vel2 = [self.get_parameter('move.line_vel_mm_s').value,
                    self.get_parameter('move.line_vel_deg_s').value]
            acc2 = [self.get_parameter('move.line_acc_mm_s2').value,
                    self.get_parameter('move.line_acc_deg_s2').value]
            success = self._move_line(goal_handle, pos6, vel2, acc2)
        else:
            self.get_logger().error('_call_move_service: named_target 또는 target_pose가 필요합니다.')
            return False
        # 이동 함수가 반환된 직후에도 안전상태를 다시 확인한다 - 이동 서비스가 성공
        # 응답을 반환하는 순간과 거의 동시에 Fault가 발생하는 경합으로 인해 Action이
        # 성공 처리되지 않도록 막는다.
        if success and self.safety_state != SafetyState.NORMAL:
            self.get_logger().warn(
                f'이동 완료 직후 안전상태 비정상({self.safety_state}) 감지 - 성공으로 처리하지 않습니다.')
            return False
        return success

    def _execute_move_named(self, goal_handle):
        result = RobotTask.Result()
        if self.safety_state != SafetyState.NORMAL:
            goal_handle.abort()
            result.success = False
            result.message = f'move_named rejected - safety_state={self.safety_state}'
            return result
        try:
            success = self._safe_call(
                self._call_move_service, goal_handle=goal_handle,
                named_target=goal_handle.request.named_target, default=False)
        except Exception as exc:  # 통신 오류 등 예외 발생 시 성공을 반환하지 않는다
            self.get_logger().error(f'move_named 실행 중 예외: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = f'move_named exception: {exc}'
            return result
        if not success and goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = f'move_named({goal_handle.request.named_target}) canceled'
            return result
        if success:
            goal_handle.succeed()
            result.success = True
        else:
            goal_handle.abort()
            result.success = False
            result.message = f'move_named({goal_handle.request.named_target}) failed'
        return result

    def _execute_move_pose(self, goal_handle):
        result = RobotTask.Result()
        if self.safety_state != SafetyState.NORMAL:
            goal_handle.abort()
            result.success = False
            result.message = f'move_pose rejected - safety_state={self.safety_state}'
            return result
        try:
            success = self._safe_call(
                self._call_move_service, goal_handle=goal_handle,
                target_pose=goal_handle.request.target_pose, default=False)
        except Exception as exc:
            self.get_logger().error(f'move_pose 실행 중 예외: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = f'move_pose exception: {exc}'
            return result
        if not success and goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = 'move_pose canceled'
            return result
        if success:
            goal_handle.succeed()
            result.success = True
        else:
            goal_handle.abort()
            result.success = False
            result.message = 'move_pose failed'
        return result

    def _execute_release_and_retry(self, goal_handle):
        result = RobotTask.Result()
        if self.safety_state != SafetyState.NORMAL:
            goal_handle.abort()
            result.success = False
            result.message = f'release_and_retry rejected - safety_state={self.safety_state}'
            return result
        try:
            # RG2를 실제로 열기 직전에 취소/안전 상태를 다시 확인한다 - 그 사이 취소나
            # Fault가 발생했다면 그리퍼를 열지 않는다.
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'release_and_retry canceled before opening gripper'
                return result
            if self.safety_state != SafetyState.NORMAL:
                goal_handle.abort()
                result.success = False
                result.message = (
                    f'release_and_retry aborted before opening gripper - '
                    f'safety_state={self.safety_state}')
                return result
            self._safe_call(self.rg2_client.open)
            success = self._safe_call(
                self._call_move_service, goal_handle=goal_handle,
                named_target='watch', default=False)
        except Exception as exc:
            self.get_logger().error(f'release_and_retry 실행 중 예외: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = f'release_and_retry exception: {exc}'
            return result
        if not success and goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = 'release_and_retry canceled'
            return result
        if success:
            goal_handle.succeed()
            result.success = True
            result.message = 'released, returned to watch'
        else:
            goal_handle.abort()
            result.success = False
            result.message = 'release_and_retry failed to return to watch'
        return result

    # ---- servo_pick ----

    def _open_rt_session(self) -> bool:
        """RT 세션을 열고, 실제로 시작이 확인된 경우에만 True를 반환한다."""
        if not self.hardware_enabled:
            self.get_logger().info('[dry_run] RT 세션 오픈 생략')
            return True
        if self._doosan is None:
            raise RuntimeError('DoosanDriver가 초기화되지 않았습니다.')
        return self._doosan.open_rt_session()

    def _close_rt_session(self) -> None:
        if not self.hardware_enabled:
            self.get_logger().info('[dry_run] RT 세션 종료 생략')
            return
        if self._doosan is None:
            return
        self._doosan.close_rt_session()

    def _estimate_payload(self) -> float:
        """들어올림 직후 외부 토크로 페이로드(kg)를 추정한다.

        TODO: 토크->페이로드 환산은 로봇 자세/동역학 모델에 의존하는 캘리브레이션이
        필요하다. 임의의 환산식을 만들어 넣지 않고 미구현 상태로 남긴다
        (_safe_call이 호출부에서 기본값 0.0으로 안전하게 처리한다).
        """
        raise NotImplementedError('_estimate_payload 구현 필요 (페이로드 추정 캘리브레이션 필요)')

    def _servo_pick_tick(self):
        abort_reason = self.servo_loop.should_abort()
        if abort_reason is not None:
            return ('ABORT', abort_reason)
        if self.servo_loop.should_close():
            return ('CLOSE', None)
        return ('CONTINUE', None)

    def _on_tool_track_during_servo(self, msg):
        self.servo_loop.on_tool_track(msg)

    def _execute_servo_pick(self, goal_handle):
        result = RobotTask.Result()
        if self.safety_state != SafetyState.NORMAL:
            goal_handle.abort()
            result.success = False
            result.message = f'servo_pick rejected - safety_state={self.safety_state}'
            return result
        if self.hardware_enabled and not self.get_parameter('servo_pick.hardware_ready').value:
            goal_handle.abort()
            result.success = False
            result.message = (
                'servo_pick rejected - servo_pick.hardware_ready=false '
                '(ToolTrack 좌표계(base_link 절대좌표) -> ServoLoop 가정(TCP 오차) 변환이 '
                '아직 구현·검증되지 않아 실제 RT 속도 명령 발행을 금지합니다.)')
            return result

        request = goal_handle.request
        rt_session_open = False
        rt_confirmed = False
        servo_sub = None

        def _finish_cancel(cancel_message):
            """취소 확인 시 하드웨어 정리를 모두 시도한 뒤, 정리가 전부 성공한
            경우에만 canceled()로 마무리한다. 정지/RT 종료/subscription 제거 중
            하나라도 실패하면 취소 성공으로 가장하지 않고 FAULT로 abort한다
            (요구사항: cleanup 실패 시 취소 성공으로 가장하지 않음)."""
            nonlocal servo_sub, rt_session_open
            stop_ok = self._cleanup_stop_motion()
            sub_ok = self._cleanup_destroy_subscription(servo_sub)
            servo_sub = None
            rt_ok = True
            if rt_session_open:
                rt_ok = self._cleanup_close_rt_session()
                rt_session_open = False
            if stop_ok and sub_ok and rt_ok:
                goal_handle.canceled()
                result.success = False
                result.message = cancel_message
                return result
            detail = (
                f'servo_pick cleanup 실패로 안전을 위해 FAULT 처리 '
                f'(move_stop={stop_ok}, subscription={sub_ok}, rt_session={rt_ok})')
            self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
            goal_handle.abort()
            result.success = False
            result.message = detail
            return result

        def _finish_shutdown():
            """while rclpy.ok()가 False가 되어(프로세스 종료 중) 루프가 끝난
            경우에만 호출된다(break로 CLOSE 상태에 도달한 경우는 호출되지 않는다 -
            while/else 참고). 정상 CLOSE/파지 조건으로 간주하지 않고, RG2를 절대
            열거나 닫지 않은 채 실제 정지 시도 및 하드웨어 정리 후 abort한다."""
            nonlocal servo_sub, rt_session_open
            self._cleanup_stop_motion()
            self._cleanup_destroy_subscription(servo_sub)
            servo_sub = None
            if rt_session_open:
                self._cleanup_close_rt_session()
                rt_session_open = False
            goal_handle.abort()
            result.success = False
            result.message = 'servo_pick aborted - rclpy가 종료 중입니다.'
            return result

        try:
            rt_session_open = True  # 오픈을 시도하는 시점부터 - 실패해도 finally에서 정리한다.
            rt_confirmed = self._safe_call(self._open_rt_session, default=False)
            if self.hardware_enabled and not rt_confirmed:
                goal_handle.abort()
                result.success = False
                result.message = 'servo_pick aborted - RT 세션 시작을 확인하지 못했습니다.'
                return result

            self.servo_loop.start(request.tool_class, request.grasp_width_mm, request.grasp_force_n)
            servo_sub = self.create_subscription(
                ToolTrack, '/vision/tool_track', self._on_tool_track_during_servo, 10,
                callback_group=self.sensor_callback_group)

            control_period_s = self.get_parameter('servo_pick.rt_control_period_s').value
            hardware_ready = self.get_parameter('servo_pick.hardware_ready').value
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    # 취소 순서: 정지 요청 -> RT 세션 종료/구독 해제(먼저 정리) -> canceled().
                    return _finish_cancel('servo_pick canceled')
                if self.safety_state != SafetyState.NORMAL:
                    goal_handle.abort()
                    result.success = False
                    result.message = f'servo_pick aborted - safety_state={self.safety_state}'
                    return result

                status, reason = self._servo_pick_tick()
                feedback = RobotTask.Feedback()
                feedback.state = self.servo_loop.get_state()
                goal_handle.publish_feedback(feedback)

                if status == 'ABORT':
                    goal_handle.abort()
                    result.success = False
                    result.message = reason
                    return result
                if status == 'CLOSE':
                    break

                cmd = self.servo_loop.step()
                # 삼중 게이트: hardware_enabled(하드웨어 모드) + hardware_ready(좌표 변환
                # 검증 완료) + rt_confirmed(StartRtControl 성공 확인) 모두 참일 때만 실제
                # RT 속도 명령을 발행한다.
                if (self.hardware_enabled and hardware_ready and rt_confirmed
                        and self._doosan is not None):
                    self._safe_call(self._doosan.publish_speedl_rt, cmd)
                time.sleep(control_period_s)
            else:
                # while의 조건(rclpy.ok())이 거짓이 되어 루프가 끝난 경우에만 실행된다
                # (break로 빠져나온 경우는 실행되지 않는다) - 즉 CLOSE 상태 도달이 아니라
                # 프로세스 종료 중이라는 뜻이다. 정상 종료로 간주하지 않는다.
                return _finish_shutdown()

            # RG2를 실제로 닫기(파지) 직전에 취소/안전 상태를 다시 확인한다.
            if goal_handle.is_cancel_requested:
                return _finish_cancel('servo_pick canceled before closing gripper')
            if self.safety_state != SafetyState.NORMAL:
                goal_handle.abort()
                result.success = False
                result.message = (
                    f'servo_pick aborted before closing gripper - safety_state={self.safety_state}')
                return result

            self._safe_call(self.rg2_client.close, request.grasp_width_mm, request.grasp_force_n)
            width_mm, grip_detected = self._safe_call(
                self.rg2_client.get_state, default=(0.0, False))
            payload_kg = self._safe_call(self._estimate_payload, default=0.0)

            # succeed() 전에 반드시 subscription 제거 + RT 세션 종료를 시도하고, 모두
            # 성공한 경우에만 성공으로 마무리한다 - cleanup 실패를 숨기고 성공으로
            # 보고하지 않는다.
            sub_ok = self._cleanup_destroy_subscription(servo_sub)
            servo_sub = None
            rt_ok = True
            if rt_session_open:
                rt_ok = self._cleanup_close_rt_session()
                rt_session_open = False

            if not (sub_ok and rt_ok):
                detail = (
                    f'servo_pick 파지 후 cleanup 실패로 안전을 위해 FAULT 처리 '
                    f'(subscription={sub_ok}, rt_session={rt_ok})')
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                goal_handle.abort()
                result.success = False
                result.message = detail
                return result

            goal_handle.succeed()
            result.success = True
            result.measured_payload_kg = payload_kg
            result.final_width_mm = width_mm
            result.grip_detected = grip_detected
            return result
        except Exception as exc:  # 통신 오류/예외 시 성공을 반환하지 않는다
            self.get_logger().error(f'servo_pick 실행 중 예외: {exc}')
            if goal_handle.is_cancel_requested:
                return _finish_cancel(f'servo_pick canceled after exception: {exc}')
            stop_ok = self._cleanup_stop_motion()
            if not stop_ok:
                self._declare_fault(
                    f'{FaultPrefix.FAULT}servo_pick 예외 처리 중 MoveStop cleanup 실패: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = f'servo_pick exception: {exc}'
            return result
        finally:
            # 위 경로들에서 이미 정리됐다면 servo_sub/rt_session_open이 None/False로
            # 남아 있어 아래는 아무 것도 하지 않는다. 이 wrapper들은 예외를 던지지
            # 않으므로 finally에서 실패하더라도 위에서 이미 결정된 result를 덮어쓰지
            # 않는다.
            if servo_sub is not None:
                self._cleanup_destroy_subscription(servo_sub)
            if rt_session_open:
                self._cleanup_close_rt_session()

    # ---- handover_hold ----

    def _enable_compliance(self) -> None:
        if not self.hardware_enabled:
            self.get_logger().info('[dry_run] compliance 모드 on 생략')
            return
        if self._doosan is None:
            raise RuntimeError('DoosanDriver가 초기화되지 않았습니다.')
        self._doosan.enable_compliance()

    def _disable_compliance(self) -> None:
        if not self.hardware_enabled:
            self.get_logger().info('[dry_run] compliance 모드 off 생략')
            return
        if self._doosan is None:
            return
        self._doosan.disable_compliance()

    def _is_pull_detected(self, robot_state) -> bool:
        """robot_state의 tool_force에서 전달 방향(pull_axis_index) 성분만 확인해
        판정한다. 다른 축의 힘/토크는 무시하므로, handover_hold 중 임의 방향의
        접촉을 당김으로 오판하지 않는다 (요구사항: 전달 방향의 당김만 정상 전달).

        pull_axis_index는 tool_force의 힘 성분(0=x,1=y,2=z)만 허용한다. 3~5(모멘트,
        Nm)는 힘 임계값(pull_force_threshold_n, N)과 단위가 달라 비교 대상이 아니므로
        허용하지 않는다.

        TODO: pull_axis_index/pull_direction_sign/pull_force_threshold_n은 실제
        그리퍼-TCP 장착 방향과 전달 자세에 따라 달라지는 캘리브레이션 값이다.
        하드웨어 셋업 전에는 임의로 축을 추측하지 않기 위해 기본값을 -1(미설정)로
        두었고, 이 경우 항상 False를 반환해 오탐(당김 오판)을 방지한다.
        """
        if not isinstance(robot_state, dict):
            return False
        axis = int(self.get_parameter('handover_hold.pull_axis_index').value)
        if axis < 0 or axis > 2:
            self.get_logger().warn(
                'handover_hold.pull_axis_index 미설정(또는 모멘트 축 지정) - '
                '당김 감지를 비활성화합니다 (힘 성분 0,1,2만 허용).')
            return False
        tool_force = robot_state.get('tool_force')
        if not tool_force:
            return False
        sign = self.get_parameter('handover_hold.pull_direction_sign').value
        threshold_n = self.get_parameter('handover_hold.pull_force_threshold_n').value
        component = sign * tool_force[axis]
        return component > threshold_n

    @staticmethod
    def _is_fresh_robot_state(robot_state, since_monotonic: float, max_age_s: float) -> bool:
        """robot_state 샘플이 since_monotonic(예: handover_hold 시작 시각) 이후에
        수신됐고, 수신된 지 max_age_s 이내인 경우에만 신선하다고 판단한다.
        handover_hold 시작 전의 오래된 샘플을 당김 판정에 사용하지 않기 위함이다."""
        if not isinstance(robot_state, dict):
            return False
        received_at = robot_state.get('received_at')
        if received_at is None or received_at < since_monotonic:
            return False
        return (time.monotonic() - received_at) <= max_age_s

    def _execute_handover_hold(self, goal_handle):
        result = RobotTask.Result()
        if self.safety_state != SafetyState.NORMAL:
            goal_handle.abort()
            result.success = False
            result.message = f'handover_hold rejected - safety_state={self.safety_state}'
            return result
        compliance_on = False

        def _finish_cancel(cancel_message):
            """취소 확인 시 하드웨어 정리를 모두 시도한 뒤, 정리가 전부 성공한
            경우에만 canceled()로 마무리한다. 실패 시 취소 성공으로 가장하지 않고
            FAULT로 abort한다 (RG2는 이 시점까지 열지 않았으므로 그대로 유지된다)."""
            nonlocal compliance_on
            stop_ok = self._cleanup_stop_motion()
            compliance_ok = True
            if compliance_on:
                compliance_ok = self._cleanup_disable_compliance()
                compliance_on = False
            if stop_ok and compliance_ok:
                goal_handle.canceled()
                result.success = False
                result.message = cancel_message
                return result
            detail = (
                f'handover_hold cleanup 실패로 안전을 위해 FAULT 처리 '
                f'(move_stop={stop_ok}, compliance={compliance_ok})')
            self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
            goal_handle.abort()
            result.success = False
            result.message = detail
            return result

        def _finish_shutdown():
            """while rclpy.ok()가 False가 되어(프로세스 종료 중) 루프가 끝난
            경우에만 호출된다(break로 pull 확정에 도달한 경우는 호출되지 않는다 -
            while/else 참고). 정상 pull 확정으로 간주하지 않고, RG2를 절대 열지
            않은 채 실제 정지 시도 및 compliance 해제 후 abort한다."""
            nonlocal compliance_on
            self._cleanup_stop_motion()
            if compliance_on:
                self._cleanup_disable_compliance()
                compliance_on = False
            goal_handle.abort()
            result.success = False
            result.message = 'handover_hold aborted - rclpy가 종료 중입니다.'
            return result

        try:
            self._safe_call(self._enable_compliance)
            compliance_on = True
            poll_interval_s = self.get_parameter('handover_hold.poll_interval_s').value
            max_age_s = self.get_parameter('handover_hold.force_sample_max_age_s').value
            confirm_needed = max(
                1, int(self.get_parameter('handover_hold.pull_confirm_samples').value))
            hold_start_time = time.monotonic()
            pull_confirm_count = 0
            # 마지막으로 실제 평가에 반영한 sample_seq. 같은 sample_seq를 여러 번
            # 읽어도(handover_hold의 폴링 주기가 상태 폴링 주기보다 빠른 경우) 한 번만
            # 평가한다 (요구사항 1: "동일한 센서 샘플은 몇 번 반복 조회해도 1회만 평가").
            last_evaluated_sample_seq = None
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    # 취소 순서: 정지 요청 -> compliance 해제(먼저 정리) -> RG2는 열지
                    # 않음 -> canceled().
                    return _finish_cancel('handover_hold canceled')
                if self.safety_state != SafetyState.NORMAL:
                    # Fault 발생 시에도 그리퍼를 자동으로 열지 않는다 (낙하 방지).
                    goal_handle.abort()
                    result.success = False
                    result.message = f'handover_hold aborted - safety_state={self.safety_state}'
                    return result

                state = self._latest_robot_state
                is_fresh = self._is_fresh_robot_state(state, hold_start_time, max_age_s)
                sample_seq = state.get('sample_seq') if isinstance(state, dict) else None

                if not is_fresh:
                    # stale이거나 handover_hold 시작 이전 샘플만 있음 - 긴 센서 단절
                    # 이전의 count가 단절 이후로 이어지지 않도록 0으로 초기화한다.
                    pull_confirm_count = 0
                    last_evaluated_sample_seq = None
                elif sample_seq is not None and sample_seq != last_evaluated_sample_seq:
                    last_evaluated_sample_seq = sample_seq
                    if self._safe_call(self._is_pull_detected, state, default=False):
                        pull_confirm_count += 1
                    else:
                        pull_confirm_count = 0
                # else: 같은 sample_seq의 신선한 중복 조회 - count를 변경하지 않는다.
                if pull_confirm_count >= confirm_needed:
                    break
                time.sleep(poll_interval_s)
            else:
                # while의 조건(rclpy.ok())이 거짓이 되어 루프가 끝난 경우에만 실행된다
                # (break로 pull 확정에 도달한 경우는 실행되지 않는다) - 프로세스 종료
                # 중이라는 뜻이므로 정상 pull 확정으로 간주하지 않는다.
                return _finish_shutdown()

            # RG2를 실제로 열기 직전에 취소/안전 상태를 다시 확인한다.
            if goal_handle.is_cancel_requested:
                return _finish_cancel('handover_hold canceled before opening gripper')
            if self.safety_state != SafetyState.NORMAL:
                goal_handle.abort()
                result.success = False
                result.message = (
                    f'handover_hold aborted before opening gripper - safety_state={self.safety_state}')
                return result

            # RG2를 열기 전에 compliance를 먼저 해제한다 - 해제가 실패하면 RG2를 열지
            # 않고 FAULT로 abort한다 (요구사항: compliance 해제 실패 시 RG2 open 금지).
            compliance_ok = True
            if compliance_on:
                compliance_ok = self._cleanup_disable_compliance()
                compliance_on = False
            if not compliance_ok:
                detail = (
                    'handover_hold compliance 해제 실패로 안전을 위해 FAULT 처리 - '
                    'RG2를 열지 않았습니다.')
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                goal_handle.abort()
                result.success = False
                result.message = detail
                return result

            # compliance 해제를 기다리는 동안 취소/Fault가 새로 발생했을 수 있으므로
            # RG2를 열기 직전에 다시 한번 확인한다.
            if goal_handle.is_cancel_requested:
                self._cleanup_stop_motion()
                goal_handle.canceled()
                result.success = False
                result.message = 'handover_hold canceled after compliance released'
                return result
            if self.safety_state != SafetyState.NORMAL:
                goal_handle.abort()
                result.success = False
                result.message = (
                    f'handover_hold aborted after compliance released - '
                    f'safety_state={self.safety_state}')
                return result

            self._safe_call(self.rg2_client.open)
            goal_handle.succeed()
            result.success = True
            result.message = 'pull_detected, released'
            return result
        except Exception as exc:
            self.get_logger().error(f'handover_hold 실행 중 예외: {exc}')
            if goal_handle.is_cancel_requested:
                return _finish_cancel(f'handover_hold canceled after exception: {exc}')
            stop_ok = self._cleanup_stop_motion()
            if not stop_ok:
                self._declare_fault(
                    f'{FaultPrefix.FAULT}handover_hold 예외 처리 중 MoveStop cleanup 실패: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = f'handover_hold exception: {exc}'
            return result
        finally:
            if compliance_on:
                self._cleanup_disable_compliance()

    # ---- fault / robot state polling ----

    def _read_robot_state(self):
        # received_at(수신 시각)을 기록해 handover_hold가 시작되기 전의 오래된
        # 샘플을 당김 판정에 사용하지 않도록 한다 (_is_fresh_robot_state 참고).
        # sample_seq는 이 함수가 실제로 새 샘플을 얻을 때마다 1씩 증가하는 식별자다 -
        # handover_hold의 폴링 주기(handover_hold.poll_interval_s)가 상태 폴링 주기
        # (safety.state_poll_period_s)보다 빠를 수 있어, 같은 물리 샘플을
        # self._latest_robot_state에서 여러 번 읽더라도 sample_seq가 같으면 중복
        # 평가하지 않도록 한다 (_execute_handover_hold 참고).
        if not self.hardware_enabled:
            self._robot_state_seq += 1
            return {
                'robot_state': DoosanRobotState.STANDBY,
                'ext_torque': [0.0] * 6,
                'tool_force': [0.0] * 6,
                'received_at': time.monotonic(),
                'sample_seq': self._robot_state_seq,
            }
        if self._doosan is None:
            return None
        robot_state = self._doosan.get_robot_state()
        if robot_state is None:
            return None
        ext_torque = self._doosan.get_external_torque() or [0.0] * 6
        tool_force = self._doosan.get_tool_force(
            ref=self.get_parameter('handover_hold.ref').value) or [0.0] * 6
        self._robot_state_seq += 1
        return {
            'robot_state': robot_state, 'ext_torque': ext_torque, 'tool_force': tool_force,
            'received_at': time.monotonic(), 'sample_seq': self._robot_state_seq,
        }

    def _check_fault(self, robot_state):
        if not isinstance(robot_state, dict):
            return None
        code = robot_state.get('robot_state')
        if code == DoosanRobotState.EMERGENCY_STOP:
            return f'{FaultPrefix.EMERGENCY_STOP}물리 비상정지(E-Stop)가 감지되었습니다 (robot_state={code}).'
        if code in (DoosanRobotState.SAFE_STOP, DoosanRobotState.SAFE_STOP2,
                    DoosanRobotState.SAFE_OFF, DoosanRobotState.SAFE_OFF2):
            return f'{FaultPrefix.PROTECTIVE_STOP}보호정지 상태가 감지되었습니다 (robot_state={code}).'
        ext_torque = robot_state.get('ext_torque') or []
        threshold = self.get_parameter('safety.external_torque_threshold_nm').value
        if ext_torque and max(abs(t) for t in ext_torque) > threshold:
            peak = max(abs(t) for t in ext_torque)
            return f'{FaultPrefix.FAULT}예상하지 못한 외력이 감지되었습니다 (ext_torque peak={peak:.1f} Nm).'
        return None

    @staticmethod
    def _classify_fault_level(fault_reason: str) -> str:
        """/robot/fault 접두어로 안전상태 단계를 분류한다 (task_manager의
        TaskManagerNode._classify_fault_prefix와 합의된 접두어)."""
        if fault_reason.startswith(FaultPrefix.EMERGENCY_STOP):
            return SafetyState.EMERGENCY_STOP
        if fault_reason.startswith(FaultPrefix.PROTECTIVE_STOP):
            return SafetyState.PROTECTIVE_STOP
        return SafetyState.FAULT

    def _declare_fault(self, fault_reason: str) -> None:
        """안전상태를 fault_reason에 따라 갱신하고 /robot/fault를 발행한다.

        SAFETY_STATE_PRIORITY상 현재 상태보다 낮은 단계로는 절대 강등하지 않는다 -
        예를 들어 이미 EMERGENCY_STOP인데 cleanup 실패로 인한 FAULT 선언 요청이
        들어와도 EMERGENCY_STOP을 덮어쓰지 않는다(요구사항: "EMERGENCY_STOP을 낮은
        상태로 자동 강등하지 않음"). 정지 시도는 예외에 안전한 _cleanup_stop_motion을
        사용한다 - task_manager의 취소 응답을 기다리지 않고 robot_control이 먼저
        정지를 시도한다."""
        new_state = self._classify_fault_level(fault_reason)
        if SAFETY_STATE_PRIORITY[new_state] < SAFETY_STATE_PRIORITY[self.safety_state]:
            return
        self.safety_state = new_state
        self._last_fault_reason = fault_reason
        self._cleanup_stop_motion()
        msg = String()
        msg.data = fault_reason
        self.pub_fault.publish(msg)

    def _on_state_poll_timer(self):
        """로봇 상태 폴링은 이미 비정상 상태여도 절대 멈추지 않는다 - 그래야 더 높은
        단계의 Fault(예: PROTECTIVE_STOP 도중 물리 E-Stop이 눌리는 경우)를 계속
        감지해 즉시 반영할 수 있다. 다만 완전히 동일한 fault_reason 문자열이 반복되면
        (예: 매 0.1초 polling마다 같은 조건이 계속 감지되는 경우) 재발행하지 않는다.
        새로 감지된 상태가 현재보다 낮은 단계라면 _declare_fault가 강등을 막는다."""
        state = self._safe_call(self._read_robot_state, default=None)
        if state is None:
            return
        self._latest_robot_state = state
        fault_reason = self._safe_call(self._check_fault, state, default=None)
        if fault_reason is None:
            return  # 새로 감지된 이상 없음 - 기존 안전상태를 자동으로 되돌리지 않는다.
        if fault_reason == self._last_fault_reason:
            return  # 완전히 동일한 메시지의 반복 발행 방지
        self._declare_fault(fault_reason)

    def _on_gripper_timer(self):
        width_mm, grip_detected = self._safe_call(
            self.rg2_client.get_state, default=(0.0, False))
        msg = GripperState()
        msg.width_mm = width_mm
        msg.grip_detected = grip_detected
        self.pub_gripper_state.publish(msg)

    # ---- /robot/recover ----

    def _on_recover(self, request, response):
        if self.safety_state == SafetyState.NORMAL:
            response.success = True
            response.message = '이미 정상 상태입니다.'
            return response

        if not self.hardware_enabled:
            if self.safety_state == SafetyState.EMERGENCY_STOP:
                response.success = False
                response.message = '[dry_run] 물리 E-Stop 상태는 소프트웨어로 복구할 수 없습니다.'
                return response
            self.safety_state = SafetyState.NORMAL
            response.success = True
            response.message = '[dry_run] 복구되었습니다.'
            return response

        if self._doosan is None:
            response.success = False
            response.message = 'DoosanDriver가 초기화되지 않았습니다.'
            return response

        robot_state = self._doosan.get_robot_state()
        if robot_state == DoosanRobotState.EMERGENCY_STOP:
            response.success = False
            response.message = '물리 E-Stop이 눌려 있습니다. 소프트웨어 복구를 거절합니다.'
            return response

        if robot_state == DoosanRobotState.STANDBY:
            # Doosan 자체 안전정지는 아니지만 소프트웨어(외력 이상 등)로 FAULT를 선언한
            # 경우 - 외력이 실제로 정상 범위로 돌아왔는지 재확인한 뒤에만 복구를 허용한다.
            ext_torque = self._doosan.get_external_torque()
            threshold = self.get_parameter('safety.external_torque_threshold_nm').value
            if ext_torque is not None and max(abs(t) for t in ext_torque) <= threshold:
                self.safety_state = SafetyState.NORMAL
                response.success = True
                response.message = f'복구 완료 (robot_state={robot_state})'
            else:
                response.success = False
                response.message = f'복구 조건 미충족 - 외력이 여전히 높습니다 (robot_state={robot_state}).'
            return response

        control_code = {
            DoosanRobotState.SAFE_STOP: DoosanRobotControl.CONTROL_RESET_SAFET_STOP,
            DoosanRobotState.SAFE_OFF: DoosanRobotControl.CONTROL_RESET_SAFET_OFF,
            DoosanRobotState.SAFE_STOP2: DoosanRobotControl.CONTROL_RECOVERY_SAFE_STOP,
            DoosanRobotState.SAFE_OFF2: DoosanRobotControl.CONTROL_RECOVERY_SAFE_OFF,
        }.get(robot_state)
        if control_code is None:
            response.success = False
            response.message = f'복구할 수 없는 상태입니다 (robot_state={robot_state}).'
            return response

        # SetRobotControl의 success만으로 NORMAL을 판단하지 않는다 - 실제 GetRobotState를
        # 다시 조회해 로봇이 정말 STANDBY로 돌아왔는지 확인한 뒤에만 복구를 인정한다.
        self._doosan.set_robot_control(control_code)
        new_state = self._doosan.get_robot_state()
        if new_state == DoosanRobotState.STANDBY:
            self.safety_state = SafetyState.NORMAL
            response.success = True
            response.message = f'복구 완료 (robot_state={robot_state} -> {new_state})'
            return response
        if new_state == DoosanRobotState.RECOVERY:
            # SAFE_STOP2/SAFE_OFF2 -> RECOVERY_* 제어는 RECOVERY 상태로만 전이시킨다
            # (SetRobotControl.srv 주석 참고). STANDBY로 되돌리려면 추가로
            # CONTROL_RESET_RECOVERY(7)가 필요한데, 이는 더 심각한 안전정지였던
            # 경로이므로 안전상 자동으로 연쇄 호출하지 않는다 - 확인 없이는 NORMAL로
            # 바꾸지 않는다.
            response.success = False
            response.message = (
                f'RECOVERY 상태로 전이되었으나 STANDBY 확인 전입니다 (robot_state={new_state}). '
                '추가 복구 단계(CONTROL_RESET_RECOVERY)가 필요하며, 자동으로 수행하지 않습니다.')
            return response
        response.success = False
        response.message = f'복구 실패 - 여전히 안전정지 상태입니다 (robot_state={new_state}).'
        return response

    # ---- action dispatch ----

    def _execute_callback(self, goal_handle):
        self._active_goal_handle = goal_handle
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
            self._active_goal_handle = None
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
