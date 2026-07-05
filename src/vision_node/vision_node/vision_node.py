import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from handover_interfaces.msg import ToolTrack, DetectionArray
from handover_interfaces.srv import SetVisionMode

from vision_node.tracking import (
    ToolTracker, pixel_to_camera_xyz, transform_to_matrix, camera_to_base, is_approaching,
)
from vision_node.hand_tracking import create_hands_detector, detect_hand_wrist_pixel


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.mode = SetVisionMode.Request.OFF
        self.tool_class = ''

        self.declare_parameter('vision.min_z_m', 0.10)
        self.declare_parameter('vision.approach_ref_x', 0.0)
        self.declare_parameter('vision.approach_ref_y', 0.0)
        self.declare_parameter('vision.tracker_alpha', 0.6)
        self.declare_parameter('vision.tracker_beta', 0.3)
        self.min_z_m = self.get_parameter('vision.min_z_m').value
        self.approach_ref_xy = (
            self.get_parameter('vision.approach_ref_x').value,
            self.get_parameter('vision.approach_ref_y').value,
        )

        self._bridge = CvBridge()
        self.tracker = ToolTracker(
            alpha=self.get_parameter('vision.tracker_alpha').value,
            beta=self.get_parameter('vision.tracker_beta').value)
        self._hands_detector = None  # 지연 생성 (TRACK_HAND 최초 진입 시)

        self.pub_tool_track = self.create_publisher(ToolTrack, '/vision/tool_track', 10)
        self.pub_hand_pose = self.create_publisher(PoseStamped, '/vision/hand_pose', 10)
        self.srv_set_mode = self.create_service(SetVisionMode, '/vision/set_mode', self._on_set_mode)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_color = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self.sub_detections = message_filters.Subscriber(
            self, DetectionArray, '/detection/tool_boxes')
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info, self.sub_detections],
            queue_size=10, slop=0.05)
        self._sync.registerCallback(self._on_synced_images)

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _on_set_mode(self, request, response):
        self.mode = request.mode
        self.tool_class = request.tool_class
        if request.mode == SetVisionMode.Request.TRACK_TOOL:
            self.tracker.reset()
        response.success = True
        response.message = f'mode set to {request.mode} (tool_class={request.tool_class})'
        return response

    def _on_synced_images(self, color_msg, depth_msg, info_msg, detection_msg):
        try:
            tf_at_stamp = self.tf_buffer.lookup_transform(
                'base_link', color_msg.header.frame_id, color_msg.header.stamp,
                timeout=Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup failed: {ex}')
            return

        if self.mode == SetVisionMode.Request.TRACK_TOOL:
            track = self._safe_call(
                self._track_tool, color_msg, depth_msg, info_msg, detection_msg,
                tf_at_stamp, self.tool_class, default=None)
            if track is not None:
                self.pub_tool_track.publish(track)
        elif self.mode == SetVisionMode.Request.TRACK_HAND:
            hand_pose = self._safe_call(
                self._track_hand, color_msg, depth_msg, info_msg, tf_at_stamp, default=None)
            if hand_pose is not None:
                self.pub_hand_pose.publish(hand_pose)

    def _tf_matrix(self, tf_at_stamp):
        t = tf_at_stamp.transform.translation
        r = tf_at_stamp.transform.rotation
        return transform_to_matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    def _track_tool(self, color_msg, depth_msg, info_msg, detection_msg, tf_at_stamp, tool_class):
        """저해상도 검출(팀원3 제공) + 3D 복원(tf_at_stamp 사용) + 알파-베타 필터로 ToolTrack을 만든다."""
        fx, fy, ppx, ppy = (float(info_msg.k[0]), float(info_msg.k[4]),
                            float(info_msg.k[2]), float(info_msg.k[5]))
        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        tf_matrix = self._tf_matrix(tf_at_stamp)

        def reconstruct(cx, cy):
            px, py = int(cx), int(cy)
            if not (0 <= py < depth_image.shape[0] and 0 <= px < depth_image.shape[1]):
                return None
            depth_m = float(depth_image[py, px]) / 1000.0
            depth_valid = depth_m >= self.min_z_m
            z = depth_m if depth_valid else (self.tracker.last_valid_z or 0.0)
            cam_xyz = pixel_to_camera_xyz(px, py, z, fx, fy, ppx, ppy)
            base_xyz = camera_to_base(cam_xyz, tf_matrix)
            return (base_xyz[0], base_xyz[1], base_xyz[2], depth_valid)

        stamp = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
        result = self.tracker.update(detection_msg.detections, tool_class, reconstruct, stamp)
        if result is None:
            return None

        position, velocity, depth_valid = result
        track = ToolTrack()
        track.header = color_msg.header
        track.tool_class = tool_class
        track.pose.position.x = position[0]
        track.pose.position.y = position[1]
        track.pose.position.z = position[2]
        track.pose.orientation.w = 1.0
        track.velocity.x = velocity[0]
        track.velocity.y = velocity[1]
        track.velocity.z = 0.0
        track.depth_valid = bool(depth_valid)
        track.approaching = bool(is_approaching(
            (position[0], position[1]), velocity, self.approach_ref_xy))
        track.confidence = 1.0
        return track

    def _track_hand(self, color_msg, depth_msg, info_msg, tf_at_stamp):
        """MediaPipe로 손목을 검출해 PoseStamped를 만든다."""
        if self._hands_detector is None:
            self._hands_detector = create_hands_detector()

        image = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        wrist_px = detect_hand_wrist_pixel(self._hands_detector, image)
        if wrist_px is None:
            return None

        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        px, py = wrist_px
        if not (0 <= py < depth_image.shape[0] and 0 <= px < depth_image.shape[1]):
            return None
        depth_m = float(depth_image[py, px]) / 1000.0
        if depth_m <= 0.0:
            return None

        fx, fy, ppx, ppy = (float(info_msg.k[0]), float(info_msg.k[4]),
                            float(info_msg.k[2]), float(info_msg.k[5]))
        cam_xyz = pixel_to_camera_xyz(px, py, depth_m, fx, fy, ppx, ppy)
        base_xyz = camera_to_base(cam_xyz, self._tf_matrix(tf_at_stamp))

        pose = PoseStamped()
        pose.header = color_msg.header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = base_xyz
        pose.pose.orientation.w = 1.0
        return pose


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
