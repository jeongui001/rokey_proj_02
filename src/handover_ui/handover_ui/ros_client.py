import json

import roslibpy


class RosClient:
    """rosbridge(WebSocket)에 접속해 필요한 토픽을 구독/퍼블리시하는 래퍼."""

    def __init__(self, host='localhost', port=9090):
        self.ros = roslibpy.Ros(host=host, port=port)
        self.on_task_status = None
        self.on_gripper_state = None
        self.on_fault = None
        self._command_topic = None

    def connect(self):
        self.ros.run()

    def close(self):
        self.ros.terminate()

    def is_connected(self) -> bool:
        return self.ros.is_connected

    def subscribe_all(self):
        roslibpy.Topic(self.ros, '/task/status', 'std_msgs/String').subscribe(
            self._on_task_status_raw)
        roslibpy.Topic(self.ros, '/gripper/state', 'handover_interfaces/GripperState').subscribe(
            self._on_gripper_state_raw)
        roslibpy.Topic(self.ros, '/robot/fault', 'std_msgs/String').subscribe(
            self._on_fault_raw)
        self._command_topic = roslibpy.Topic(
            self.ros, '/user_command/text', 'std_msgs/String')

    def publish_command(self, text: str):
        self._command_topic.publish(roslibpy.Message({'data': text}))

    def _on_task_status_raw(self, message):
        if self.on_task_status is None:
            return
        payload = json.loads(message['data'])
        self.on_task_status(payload.get('state', ''), payload.get('detail', ''))

    def _on_gripper_state_raw(self, message):
        if self.on_gripper_state is None:
            return
        self.on_gripper_state(message.get('width_mm', 0.0), message.get('grip_detected', False))

    def _on_fault_raw(self, message):
        if self.on_fault is None:
            return
        self.on_fault(message.get('data', ''))
