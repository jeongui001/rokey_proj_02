import math
import time

import rclpy

from robot_control.safety_monitor import DoosanRobotControl, SafetyState


class DoosanDriver:
    """dsr_msgs2 서비스와 토픽을 감싸는 하드웨어 경계."""

    def __init__(self, node):
        try:
            from dsr_msgs2.srv import (
                GetCurrentPosx,
                GetExternalTorque,
                GetRobotState,
                GetToolForce,
                MoveJoint,
                MoveLine,
                MoveStop,
                ReleaseComplianceCtrl,
                SetRobotControl,
                TaskComplianceCtrl,
            )
            from dsr_msgs2.msg import SpeedlStream
        except ImportError as exc:
            raise RuntimeError(
                'hardware_enabled=true에는 dsr_msgs2가 필요합니다. '
                'Doosan ROS2 워크스페이스를 source 하세요.'
            ) from exc

        self._node = node
        self._MoveJoint = MoveJoint
        self._MoveLine = MoveLine
        self._MoveStop = MoveStop
        self._GetRobotState = GetRobotState
        self._SetRobotControl = SetRobotControl
        self._GetExternalTorque = GetExternalTorque
        self._GetToolForce = GetToolForce
        self._GetCurrentPosx = GetCurrentPosx
        self._TaskComplianceCtrl = TaskComplianceCtrl
        self._ReleaseComplianceCtrl = ReleaseComplianceCtrl
        self._SpeedlStream = SpeedlStream

        prefix = f"/{node.get_parameter('robot_id').value}"
        group = node.hardware_callback_group
        self._cli_move_joint = node.create_client(
            MoveJoint, f'{prefix}/motion/move_joint', callback_group=group)
        self._cli_move_line = node.create_client(
            MoveLine, f'{prefix}/motion/move_line', callback_group=group)
        self._cli_move_stop = node.create_client(
            MoveStop, f'{prefix}/motion/move_stop', callback_group=group)
        self._cli_get_robot_state = node.create_client(
            GetRobotState, f'{prefix}/system/get_robot_state', callback_group=group)
        self._cli_set_robot_control = node.create_client(
            SetRobotControl, f'{prefix}/system/set_robot_control', callback_group=group)
        self._cli_get_ext_torque = node.create_client(
            GetExternalTorque, f'{prefix}/aux_control/get_external_torque',
            callback_group=group)
        self._cli_get_tool_force = node.create_client(
            GetToolForce, f'{prefix}/aux_control/get_tool_force', callback_group=group)
        self._cli_get_current_posx = node.create_client(
            GetCurrentPosx, f'{prefix}/aux_control/get_current_posx',
            callback_group=group)
        self._cli_task_compliance = node.create_client(
            TaskComplianceCtrl, f'{prefix}/force/task_compliance_ctrl',
            callback_group=group)
        self._cli_release_compliance = node.create_client(
            ReleaseComplianceCtrl, f'{prefix}/force/release_compliance_ctrl',
            callback_group=group)
        self._pub_speedl = node.create_publisher(
            SpeedlStream, f'{prefix}/speedl_stream', 10)

    @staticmethod
    def _response_success(response) -> bool:
        return bool(response is not None and response.success)

    def _wait_for_future(self, future, timeout_s):
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result() if future.done() else None

    def _call_move_with_cancel(
            self, client, request, goal_handle, poll_interval_s, timeout_s):
        if not client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().error(f'{client.srv_name} 서비스 연결 실패')
            return False
        future = client.call_async(request)
        start = time.monotonic()
        while rclpy.ok():
            if future.done():
                break
            canceled = goal_handle is not None and goal_handle.is_cancel_requested
            unsafe = self._node.safety_state != SafetyState.NORMAL
            if canceled or unsafe:
                self.stop(self._node.get_parameter('safety.fault_stop_mode').value)
                self._wait_for_future(future, 2.0)
                return False
            if time.monotonic() - start > timeout_s:
                self._node.get_logger().error('move 서비스 응답 타임아웃')
                self.stop(self._node.get_parameter('safety.fault_stop_mode').value)
                return False
            time.sleep(poll_interval_s)
        if not future.done() or self._node.safety_state != SafetyState.NORMAL:
            return False
        return self._response_success(future.result())

    def move_joint(
            self, goal_handle, pos_deg6, vel, acc, radius_mm=0.0,
            sync_type=0, poll_interval_s=0.05, timeout_s=30.0) -> bool:
        request = self._MoveJoint.Request()
        request.pos = [float(value) for value in pos_deg6]
        request.vel = float(vel)
        request.acc = float(acc)
        request.time = 0.0
        request.radius = float(radius_mm)
        request.mode = 0
        request.blend_type = 0
        request.sync_type = int(sync_type)
        return self._call_move_with_cancel(
            self._cli_move_joint, request, goal_handle, poll_interval_s, timeout_s)

    def move_line(
            self, goal_handle, pos6, vel2, acc2, ref=0, radius_mm=0.0,
            sync_type=0, poll_interval_s=0.05, timeout_s=30.0) -> bool:
        request = self._MoveLine.Request()
        request.pos = [float(value) for value in pos6]
        request.vel = [float(vel2[0]), float(vel2[1])]
        request.acc = [float(acc2[0]), float(acc2[1])]
        request.time = 0.0
        request.radius = float(radius_mm)
        request.ref = int(ref)
        request.mode = 0
        request.blend_type = 0
        request.sync_type = int(sync_type)
        return self._call_move_with_cancel(
            self._cli_move_line, request, goal_handle, poll_interval_s, timeout_s)

    def stop(self, stop_mode=1) -> bool:
        if not self._cli_move_stop.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().error('motion/move_stop 서비스 연결 실패')
            return False
        request = self._MoveStop.Request()
        request.stop_mode = int(stop_mode)
        response = self._wait_for_future(self._cli_move_stop.call_async(request), 2.0)
        return self._response_success(response)

    def get_robot_state(self):
        if not self._cli_get_robot_state.wait_for_service(timeout_sec=1.0):
            return None
        response = self._wait_for_future(
            self._cli_get_robot_state.call_async(self._GetRobotState.Request()), 2.0)
        return int(response.robot_state) if self._response_success(response) else None

    def get_external_torque(self):
        if not self._cli_get_ext_torque.wait_for_service(timeout_sec=1.0):
            return None
        response = self._wait_for_future(
            self._cli_get_ext_torque.call_async(self._GetExternalTorque.Request()), 2.0)
        return list(response.ext_torque) if self._response_success(response) else None

    def get_tool_force(self, ref=0):
        if int(ref) not in (0, 2):
            self._node.get_logger().error(
                f'get_tool_force ref={ref}: BASE=0 또는 WORLD=2만 허용')
            return None
        if not self._cli_get_tool_force.wait_for_service(timeout_sec=1.0):
            return None
        request = self._GetToolForce.Request()
        request.ref = int(ref)
        response = self._wait_for_future(self._cli_get_tool_force.call_async(request), 2.0)
        return list(response.tool_force) if self._response_success(response) else None

    def get_current_posx(self, ref=0):
        if not self._cli_get_current_posx.wait_for_service(timeout_sec=1.0):
            return None
        request = self._GetCurrentPosx.Request()
        request.ref = int(ref)
        response = self._wait_for_future(self._cli_get_current_posx.call_async(request), 2.0)
        if not self._response_success(response) or not response.task_pos_info:
            return None
        data = list(response.task_pos_info[0].data)
        if len(data) < 6:
            return None
        pos6 = [float(value) for value in data[:6]]
        return pos6 if all(math.isfinite(value) for value in pos6) else None

    def set_robot_control(self, code) -> bool:
        if not self._cli_set_robot_control.wait_for_service(timeout_sec=1.0):
            return False
        request = self._SetRobotControl.Request()
        request.robot_control = int(code)
        response = self._wait_for_future(self._cli_set_robot_control.call_async(request), 2.0)
        return self._response_success(response)

    def publish_speedl(self, command, *, accel_param_prefix, period_param_name):
        # 실제 단위가 확인될 때까지 hardware_ready 게이트로 발행을 막는다.
        # accel_param_prefix/period_param_name으로 호출자(servo_pick 또는
        # handover_approach)에 맞는 자신의 파라미터를 쓴다 - 예전에는 항상
        # servo_pick 것만 썼던 교차배선을 여기서 바로잡는다.
        message = self._SpeedlStream()
        message.vel = [
            command.vx, command.vy, command.vz, 0.0, 0.0, command.yaw_rate]
        message.acc = [
            self._node.get_parameter(f'{accel_param_prefix}.speedl_acc_trans_mm_s2').value,
            self._node.get_parameter(f'{accel_param_prefix}.speedl_acc_rot_deg_s2').value]
        message.time = self._node.get_parameter(period_param_name).value
        self._pub_speedl.publish(message)

    def enable_compliance(self):
        if not self._cli_task_compliance.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('task_compliance_ctrl 서비스 연결 실패')
        request = self._TaskComplianceCtrl.Request()
        request.stx = list(
            self._node.get_parameter('handover_hold.compliance_stiffness').value)
        request.ref = 0
        request.time = self._node.get_parameter(
            'handover_hold.compliance_transition_s').value
        response = self._wait_for_future(
            self._cli_task_compliance.call_async(request), 3.0)
        if not self._response_success(response):
            raise RuntimeError('task_compliance_ctrl 실패')

    def disable_compliance(self):
        if not self._cli_release_compliance.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('release_compliance_ctrl 서비스 연결 실패')
        response = self._wait_for_future(
            self._cli_release_compliance.call_async(
                self._ReleaseComplianceCtrl.Request()), 3.0)
        if not self._response_success(response):
            raise RuntimeError('release_compliance_ctrl 실패')


__all__ = ['DoosanDriver', 'DoosanRobotControl']
