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

    def _on_user_command(self, msg):
        pass

    def _on_tool_track(self, msg):
        pass

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
