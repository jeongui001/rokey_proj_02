"""컨테이너 안에서 카메라 이미지를 구독해 mediapipe로 손을 검출하고,
결과를 임시 토픽(/vision/hand_track_docker)에 String(JSON)으로 발행하는 검증용 노드.

실제 HandTrack 커스텀 메시지 타입은 아직 안 쓴다 - handover_interfaces를
컨테이너 안에 빌드해 넣는 건 다음 단계 작업이라, 우선 이 단계에서는
"카메라 프레임이 컨테이너까지 들어와서 mediapipe가 처리하고 결과가
나온다"는 것만 검증한다.
"""

import json

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from hand_tracking import create_hands_detector, detect_hand, is_fist


class HandTrackDockerNode(Node):

    def __init__(self):
        super().__init__('hand_track_docker_node')
        self._bridge = CvBridge()
        self._hands = create_hands_detector()
        self._pub = self.create_publisher(String, '/vision/hand_track_docker', 10)
        self.create_subscription(
            Image, '/camera/color/image_raw', self._on_image, qos_profile_sensor_data)
        self.get_logger().info('hand_track_docker_node started, waiting for images...')

    def _on_image(self, msg):
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = detect_hand(self._hands, bgr)
        if result is None:
            payload = {'detected': False}
        else:
            palm_px, landmarks, confidence = result
            payload = {
                'detected': True,
                'palm_px': list(palm_px),
                'confidence': confidence,
                'is_fist': is_fist(landmarks),
            }
        out = String()
        out.data = json.dumps(payload)
        self._pub.publish(out)
        self.get_logger().info(f'published: {out.data}')


def main():
    rclpy.init()
    node = HandTrackDockerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
