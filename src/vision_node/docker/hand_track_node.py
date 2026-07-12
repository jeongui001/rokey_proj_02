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
from rclpy.qos import (
    QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from hand_tracking import create_hands_detector, detect_hand, is_fist


class HandTrackDockerNode(Node):

    def __init__(self):
        super().__init__('hand_track_docker_node')
        self._bridge = CvBridge()
        self._hands = create_hands_detector()  # 게이트가 꺼져 있어도 디텍터는 warm하게 유지한다
        self._enabled = False
        # 게이트가 꺼져 있는 동안에는 이미지 구독 자체를 만들지 않는다. 구독만 걸어두고
        # 콜백에서 일찍 빠져나오는 방식으로는 카메라 퍼블리셔가 여전히 305KB짜리 프레임을
        # 이 컨테이너로 계속 보내야 해서(424x240 RGB, ~38Hz -> 약 11MB/s), DDS/UDP 부하가
        # 그대로 남아 같은 카메라를 보는 vision_node 쪽 프레임이 깨진다. TRACK_HAND일 때만
        # 구독을 만들고, 꺼지면 destroy_subscription으로 트래픽을 0으로 만든다.
        self._image_sub = None
        self._pub = self.create_publisher(String, '/vision/hand_track_docker', 10)
        # vision_node가 TRACK_HAND 모드일 때만 True를 보낸다. transient_local이라
        # 이 컨테이너가 vision_node보다 늦게 떠도 마지막 값을 즉시 받는다 - 기본 QoS(VOLATILE)로
        # 구독하면 래치된 값을 못 받아 손 검출이 영영 안 켜지므로 QoS를 반드시 맞춰야 한다.
        self.create_subscription(
            Bool, '/vision/hand_track_enable', self._on_enable,
            QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ))
        self.get_logger().info('hand_track_docker_node started, waiting for images...')

    def _on_enable(self, msg):
        if msg.data == self._enabled:
            return
        self._enabled = msg.data
        if self._enabled:
            self._image_sub = self.create_subscription(
                Image, '/camera/color/image_raw', self._on_image, qos_profile_sensor_data)
        elif self._image_sub is not None:
            self.destroy_subscription(self._image_sub)
            self._image_sub = None
        self.get_logger().info(
            f'hand detection {"ENABLED" if self._enabled else "DISABLED"}')

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
        self.get_logger().info(f'published: {out.data}', throttle_duration_sec=1.0)


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
