import json

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode


class State:
    IDLE = 'IDLE'
    PARSING = 'PARSING'
    MOVE_TO_WATCH = 'MOVE_TO_WATCH'
    DETECT_TRACK = 'DETECT_TRACK'
    SERVO_PICK = 'SERVO_PICK'
    VERIFY_GRASP = 'VERIFY_GRASP'
    MOVE_SAFE = 'MOVE_SAFE'
    TRACK_HAND = 'TRACK_HAND'
    WAIT_PULL = 'WAIT_PULL'
    RELEASE = 'RELEASE'
    HOME = 'HOME'
    FAULT = 'FAULT'


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager')

        self.declare_parameter('detect_track_max_cycles', 3)
        self.declare_parameter('verify_grasp_max_retries', 2)
        self.declare_parameter('wait_pull_timeout_s', 60.0)
        self.declare_parameter('hand_detect_timeout_s', 5.0)

        self.state = State.IDLE
        self.current_tool = None
        self._detect_track_cycles = 0
        self._verify_grasp_retries = 0
        self._hand_timeout_timer = None
        self._wait_pull_timeout_timer = None

        self.pub_status = self.create_publisher(String, '/task/status', 10)
        self.sub_command = self.create_subscription(
            String, '/user_command/text', self._on_user_command, 10)
        self.sub_tool_track = self.create_subscription(
            ToolTrack, '/vision/tool_track', self._on_tool_track, 10)
        self.sub_hand_pose = self.create_subscription(
            PoseStamped, '/vision/hand_pose', self._on_hand_pose, 10)
        self.sub_fault = self.create_subscription(
            String, '/robot/fault', self._on_fault, 10)

        self.set_mode_client = self.create_client(SetVisionMode, '/vision/set_mode')
        self.robot_task_client = ActionClient(self, RobotTask, 'robot_task')

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _publish_status(self, detail=''):
        msg = String()
        msg.data = json.dumps({'state': self.state, 'detail': detail})
        self.pub_status.publish(msg)

    def _set_state(self, new_state, detail=''):
        self.state = new_state
        self._publish_status(detail)

    def _on_fault(self, msg):
        if self.state == State.FAULT:
            return
        self._set_state(State.FAULT, detail=msg.data)

    def _call_llm(self, text: str) -> dict:
        """LLM API를 호출해 {"tool": ..., "action": ...}를 반환한다. 스키마 검증·재시도 포함."""
        raise NotImplementedError('_call_llm 구현 필요')

    def _on_user_command(self, msg):
        if self.state != State.IDLE:
            return
        self._set_state(State.PARSING, detail=msg.data)
        self._handle_parsing(msg.data)

    def _handle_parsing(self, text):
        parsed = self._safe_call(self._call_llm, text, default=None)
        if not parsed or 'tool' not in parsed:
            self._set_state(State.IDLE, detail='명령을 이해하지 못했습니다. 다시 말씀해주세요.')
            return
        self.current_tool = parsed['tool']
        self._detect_track_cycles = 0
        self._verify_grasp_retries = 0
        self._set_state(State.MOVE_TO_WATCH)
        self._set_vision_mode(SetVisionMode.Request.TRACK_TOOL, self.current_tool)
        self._send_robot_goal('move_named', named_target='watch')

    def _set_vision_mode(self, mode, tool_class=''):
        request = SetVisionMode.Request()
        request.mode = mode
        request.tool_class = tool_class
        self.set_mode_client.call_async(request)

    def _send_robot_goal(self, task_type, named_target='', target_pose=None,
                          tool_class='', grasp_width_mm=0.0, grasp_force_n=0.0):
        goal = RobotTask.Goal()
        goal.task_type = task_type
        goal.named_target = named_target
        if target_pose is not None:
            goal.target_pose = target_pose
        goal.tool_class = tool_class
        goal.grasp_width_mm = grasp_width_mm
        goal.grasp_force_n = grasp_force_n
        future = self.robot_task_client.send_goal_async(
            goal, feedback_callback=self._on_robot_feedback)
        future.add_done_callback(self._on_goal_response)

    def _on_robot_feedback(self, feedback_msg):
        self._publish_status(detail=f'servo:{feedback_msg.feedback.state}')

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._set_state(State.FAULT, detail='goal rejected')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_robot_result)

    def _on_robot_result(self, future):
        response = future.result()
        result = response.result
        if self.state == State.MOVE_TO_WATCH:
            self._handle_move_to_watch_result(result)
        elif self.state == State.SERVO_PICK:
            self._handle_servo_pick_result(result)
        elif self.state == State.VERIFY_GRASP:
            self._handle_release_and_retry_result(result)
        elif self.state == State.MOVE_SAFE:
            self._handle_move_safe_result(result)
        elif self.state == State.TRACK_HAND:
            self._handle_track_hand_result(result)
        elif self.state == State.WAIT_PULL:
            self._handle_wait_pull_result(result)
        elif self.state == State.RELEASE:
            self._handle_release_result(result)
        elif self.state == State.HOME:
            self._handle_home_result(result)

    def _handle_move_to_watch_result(self, result):
        if result.success:
            self._set_state(State.DETECT_TRACK)
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _check_trigger(self, tool_track_msg) -> bool:
        """시야 내 + approaching이면 True (완화된 트리거 판정, 데모.md 1.3절)."""
        raise NotImplementedError('_check_trigger 구현 필요')

    def _get_grasp_spec(self, tool_class: str):
        """(grasp_width_mm, grasp_force_n) 등록된 공구 스펙을 반환한다."""
        raise NotImplementedError('_get_grasp_spec 구현 필요')

    def _on_tool_track(self, msg):
        if self.state != State.DETECT_TRACK:
            return
        triggered = self._safe_call(self._check_trigger, msg, default=False)
        if not triggered:
            self._detect_track_cycles += 1
            max_cycles = self.get_parameter('detect_track_max_cycles').value
            if self._detect_track_cycles >= max_cycles:
                self._set_state(State.IDLE, detail='벨트에 없음')
            return
        spec = self._safe_call(self._get_grasp_spec, self.current_tool, default=None)
        width_mm, force_n = spec if spec else (0.0, 0.0)
        self._set_state(State.SERVO_PICK)
        self._send_robot_goal(
            'servo_pick', tool_class=self.current_tool,
            grasp_width_mm=width_mm, grasp_force_n=force_n)

    def _verify_grasp(self, result) -> bool:
        """무게·폭·grip_detected 삼중 확인 (데모.md 2.6/VERIFY_GRASP)."""
        raise NotImplementedError('_verify_grasp 구현 필요')

    def _handle_servo_pick_result(self, result):
        if not result.success:
            if 'torque' in result.message:
                self._set_state(State.FAULT, detail=result.message)
            else:
                self._detect_track_cycles = 0
                self._set_state(State.DETECT_TRACK, detail=result.message)
            return
        self._set_state(State.VERIFY_GRASP)
        verified = self._safe_call(self._verify_grasp, result, default=False)
        if verified:
            self._set_state(State.MOVE_SAFE)
            self._send_robot_goal('move_named', named_target='safe')
            return
        self._verify_grasp_retries += 1
        max_retries = self.get_parameter('verify_grasp_max_retries').value
        if self._verify_grasp_retries > max_retries:
            self._set_state(State.IDLE, detail='파지 검증 실패 - 보고')
            return
        self._send_robot_goal('release_and_retry')

    def _handle_release_and_retry_result(self, result):
        if result.success:
            self._detect_track_cycles = 0
            self._set_state(State.DETECT_TRACK)
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _on_hand_pose(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
