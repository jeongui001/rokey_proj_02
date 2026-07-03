import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.mode = SetVisionMode.Request.OFF
        self.tool_class = ''

        self.pub_tool_track = self.create_publisher(ToolTrack, '/vision/tool_track', 10)
        self.pub_hand_pose = self.create_publisher(PoseStamped, '/vision/hand_pose', 10)
        self.srv_set_mode = self.create_service(SetVisionMode, '/vision/set_mode', self._on_set_mode)

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _on_set_mode(self, request, response):
        self.mode = request.mode
        self.tool_class = request.tool_class
        response.success = True
        response.message = f'mode set to {request.mode} (tool_class={request.tool_class})'
        return response

    def _track_tool(self, color_msg, depth_msg, tf_at_stamp, tool_class):
        """저해상도 YOLO 검출 + 3D 복원(tf_at_stamp 사용) + 칼만/알파-베타 필터로 ToolTrack을 만든다."""
        raise NotImplementedError('_track_tool 구현 필요')

    def _track_hand(self, color_msg):
        """MediaPipe 등으로 손을 검출해 PoseStamped를 만든다."""
        raise NotImplementedError('_track_hand 구현 필요')


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
