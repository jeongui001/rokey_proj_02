"""object_detection(팀원3) 노드 - 전체 계획.md 1.4/4.2절 계약 구현.

realsense2_camera가 발행하는 컬러 이미지를 구독해 YOLO로 공구를 검출하고, 그 결과를
DetectionArray로 /detection/tool_boxes에 발행한다. 뎁스 융합·추적·3D PCA yaw 계산·베이스
변환은 vision_node(팀원2)가 이 토픽을 받아 수행한다(전체 계획.md 8절 역할 분리).

과거엔 이 노드가 pyrealsense2로 카메라를 직접 잡고 추적·yaw·베이스 변환까지 전부 혼자
했으나(하드웨어 TF 연동 전 단독 실행용 프로토타입), 그 우월 기능(3D PCA 축, patch median
뎁스 등)이 vision_node로 이식되면서 이 노드는 YOLO 추론 전담으로 축소됐다. RealSense는
한 프로세스만 USB 파이프라인을 열 수 있어 realsense2_camera 경유가 사실상 강제이기도 하다
(hand_tracking/향후 대시보드와 카메라를 공유해야 함).
"""
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from handover_interfaces.msg import Detection2D, DetectionArray

# 클래스 구분용 색상 팔레트 (BGR). cls_id 순서대로 순환 배정 - 프로토타입과 동일
CLASS_COLORS = [
    (0, 255, 0),    # 초록
    (255, 100, 0),  # 파랑-청록
    (0, 0, 255),    # 빨강
    (0, 255, 255),  # 노랑
    (255, 0, 255),  # 마젠타
    (255, 255, 0),  # 시안
]


def _default_resource(name):
    """colcon 설치 경로가 있으면 share/resource, 없으면(미빌드 개발 중) 소스 트리 경로."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('vision_node'), 'resource', name)
        if os.path.exists(path):
            return path
    except Exception:
        pass
    return os.path.join(os.path.dirname(__file__), '..', 'resource', name)


class ToolDetectionNode(Node):
    """YOLO 공구 검출 전담 노드(object_detection 역할) - DetectionArray만 발행한다."""

    def __init__(self):
        super().__init__('tool_detection_node')
        self.declare_parameter('model_path', _default_resource('tool_detector_best.pt'))
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('show_window', False)  # 개발 확인용 - YOLO bbox만 그리는 RGB 창

        self.conf = self.get_parameter('conf').value
        self.show_window = self.get_parameter('show_window').value

        from ultralytics import YOLO  # import가 느려서(수 초) 노드 초기화 시점에 수행
        self.model = YOLO(self.get_parameter('model_path').value)
        self.class_colors = {name: CLASS_COLORS[i % len(CLASS_COLORS)]
                             for i, name in self.model.names.items()}

        self._bridge = CvBridge()
        self.pub_detections = self.create_publisher(DetectionArray, '/detection/tool_boxes', 10)
        # 퍼블리셔: realsense2_camera(기성 패키지). vision_node와 동일 토픽을 구독해
        # 같은 프레임을 본다. 카메라 드라이버는 BEST_EFFORT(qos_profile_sensor_data)를 쓰도록
        # 권장되므로 기본 QoS(RELIABLE) 대신 명시적으로 맞춰 드라이버 QoS 변경에 조용히 깨지지 않게 한다.
        self.sub_color = self.create_subscription(
            Image, '/camera/color/image_raw', self._on_color, qos_profile_sensor_data)

    def _on_color(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        t0 = time.perf_counter()
        result = self.model.predict(source=frame, conf=self.conf, verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        det_msg = DetectionArray()
        # color 프레임과 동일 stamp (전체 계획.md 4.2절 계약) - vision_node의
        # ApproximateTimeSynchronizer가 이 시각으로 depth/camera_info와 맞춘다.
        det_msg.header = msg.header
        detections = []
        # pose 모델(yolo*-pose)이면 keypoints가 박스와 같은 순서로 온다. box 모델이면 None -
        # 그땐 kpt 필드가 기본값 0으로 남아 vision_node가 bbox 중심/depth-PCA로 폴백한다.
        keypoints = result.keypoints
        for i, box in enumerate(result.boxes):
            d = Detection2D()
            d.class_name = result.names[int(box.cls[0])]
            d.score = float(box.conf[0])
            d.x1, d.y1, d.x2, d.y2 = map(int, box.xyxy[0])
            if keypoints is not None and i < len(keypoints.xy):
                xy = keypoints.xy[i]  # (2, 2): [p0, p1] 픽셀 좌표
                # kpt별 conf가 없는 모델(내보내기 경로에 따라 conf=None)은 box score로 대신한다
                conf = keypoints.conf[i] if keypoints.conf is not None else None
                if len(xy) >= 2:
                    d.kpt0_x, d.kpt0_y = float(xy[0][0]), float(xy[0][1])
                    d.kpt1_x, d.kpt1_y = float(xy[1][0]), float(xy[1][1])
                    d.kpt0_conf = float(conf[0]) if conf is not None else d.score
                    d.kpt1_conf = float(conf[1]) if conf is not None else d.score
            det_msg.detections.append(d)
            detections.append(d)
        # 프로파일링: 추론 시간 + 이 노드 구간(capture->publish 직전) 지연을 실어 보낸다 -
        # vision_node가 sync 대기/e2e와 합쳐 VisionTiming(/perception/timing)으로 통합 발행.
        det_msg.infer_ms = float(infer_ms)
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now_s = self.get_clock().now().nanoseconds * 1e-9
        det_msg.detect_latency_ms = float((now_s - stamp_s) * 1000.0)
        self.pub_detections.publish(det_msg)

        if self.show_window:
            self._draw_debug(frame, detections)

    def _draw_debug(self, frame, detections):
        """YOLO bbox만 그리는 단순 RGB 창 - 뎁스/축 시각화는 vision_node 쪽에서 확인한다."""
        for d in detections:
            color = self.class_colors.get(d.class_name, (0, 255, 0))
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
            cv2.putText(frame, f'{d.class_name} {d.score * 100:.0f}%', (d.x1, d.y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow('tool_detection_node', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ToolDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
