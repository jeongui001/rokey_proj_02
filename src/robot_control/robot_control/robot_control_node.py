import time

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from handover_interfaces.action import RobotTask

from robot_control.rg2_client import RG2Client
from robot_control.servo_loop import ServoLoop


class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')

        self.declare_parameter('rg2_ip', '192.168.1.1')
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

        self.rg2_client = RG2Client(ip=self.get_parameter('rg2_ip').value)
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
        )

        self._action_server = ActionServer(
            self, RobotTask, 'robot_task', execute_callback=self._execute_callback)

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    # ---- move / place_down / release_and_retry ----

    def _call_move_service(self, named_target='', target_pose=None) -> bool:
        """Doosan 모션 서비스(정적 이동) 호출. dsr_msgs2 등 드라이버 서비스 인터페이스 확인 후 구현."""
        raise NotImplementedError('_call_move_service 구현 필요')

    def _execute_move_named(self, goal_handle):
        result = RobotTask.Result()
        success = self._safe_call(
            self._call_move_service, named_target=goal_handle.request.named_target, default=False)
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
        success = self._safe_call(
            self._call_move_service, target_pose=goal_handle.request.target_pose, default=False)
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
        self._safe_call(self.rg2_client.open)
        success = self._safe_call(self._call_move_service, named_target='watch', default=False)
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

    def _open_rt_session(self) -> None:
        """Doosan 실시간 제어 세션을 연다. 드라이버 RT API 확인 후 구현."""
        raise NotImplementedError('_open_rt_session 구현 필요')

    def _close_rt_session(self) -> None:
        """실시간 제어 세션을 닫고 서비스 모션 모드로 복귀한다."""
        raise NotImplementedError('_close_rt_session 구현 필요')

    def _estimate_payload(self) -> float:
        """들어올림 직후 외부 토크로 페이로드(kg)를 추정한다."""
        raise NotImplementedError('_estimate_payload 구현 필요')

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
        from handover_interfaces.msg import ToolTrack

        request = goal_handle.request
        result = RobotTask.Result()

        self._safe_call(self._open_rt_session)
        self.servo_loop.start(request.tool_class, request.grasp_width_mm, request.grasp_force_n)
        servo_sub = self.create_subscription(
            ToolTrack, '/vision/tool_track', self._on_tool_track_during_servo, 10)

        try:
            while rclpy.ok():
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

                self.servo_loop.step()
                time.sleep(0.01)

            self._safe_call(self.rg2_client.close, request.grasp_width_mm, request.grasp_force_n)
            width_mm, grip_detected = self._safe_call(
                self.rg2_client.get_state, default=(0.0, False))
            payload_kg = self._safe_call(self._estimate_payload, default=0.0)

            goal_handle.succeed()
            result.success = True
            result.measured_payload_kg = payload_kg
            result.final_width_mm = width_mm
            result.grip_detected = grip_detected
        finally:
            self.destroy_subscription(servo_sub)
            self._safe_call(self._close_rt_session)

        return result

    # ---- action dispatch ----

    def _execute_callback(self, goal_handle):
        task_type = goal_handle.request.task_type
        handlers = {
            'move_named': self._execute_move_named,
            'move_pose': self._execute_move_pose,
            'place_down': self._execute_move_named,
            'release_and_retry': self._execute_release_and_retry,
            'servo_pick': self._execute_servo_pick,
        }
        handler = handlers.get(task_type)
        if handler is None:
            goal_handle.abort()
            result = RobotTask.Result()
            result.success = False
            result.message = f'unknown task_type: {task_type}'
            return result
        return handler(goal_handle)


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
