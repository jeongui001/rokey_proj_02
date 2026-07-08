"""팀원3의 object_detection 노드가 아직 없을 때, /detection/tool_boxes에
가짜 bbox를 흘려보내 vision_node를 독립적으로 검증하기 위한 개발용 노드."""

import rclpy
from rclpy.node import Node

from handover_interfaces.msg import DetectionArray, Detection2D


class FakeDetectionPublisher(Node):
    def __init__(self):
        super().__init__('fake_detection_publisher')
        self.declare_parameter('tool_class', 'spanner')
        self.declare_parameter('rate_hz', 30.0)
        # stationary=true면 왕복 없이 (cx, cy)에 고정 - approaching/velocity 노이즈 없이
        # 축(yaw) 계산만 순수하게 눈으로 확인하고 싶을 때 사용 (실물을 그 자리에 놓고 돌리면 됨).
        self.declare_parameter('stationary', False)
        self.declare_parameter('cx', 212.0)
        self.declare_parameter('cy', 120.0)
        self.tool_class = self.get_parameter('tool_class').value
        self.stationary = self.get_parameter('stationary').value
        self.cx0 = self.get_parameter('cx').value
        self.cy0 = self.get_parameter('cy').value
        rate_hz = self.get_parameter('rate_hz').value

        self.pub = self.create_publisher(DetectionArray, '/detection/tool_boxes', 10)
        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        self._t = 0.0
        self._dt = 1.0 / rate_hz

    def _on_timer(self):
        if self.stationary:
            cx = self.cx0
        else:
            # (cx0, cy0) 중심으로 좌우 왕복하는 고정 크기 bbox
            cx = self.cx0 + 100.0 * ((self._t % 4.0) / 2.0 - 1.0)
        msg = DetectionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        det = Detection2D()
        det.class_name = self.tool_class
        det.score = 0.95
        det.x1, det.y1 = int(cx - 15), int(self.cy0 - 10)
        det.x2, det.y2 = int(cx + 15), int(self.cy0 + 10)
        msg.detections = [det]
        self.pub.publish(msg)
        self._t += self._dt


def main(args=None):
    rclpy.init(args=args)
    node = FakeDetectionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
