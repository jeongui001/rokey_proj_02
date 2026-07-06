"""rclpy로 ROS2에 직접 접속해 handover_ui가 필요한 토픽을 구독/퍼블리시한다.

원래는 rosbridge_server(WebSocket) + roslibpy로 접속했으나(브라우저 UI 초안의
흔적), UI가 PyQt 데스크톱 앱으로 확정된 뒤에는 rclpy로 직접 붙는 것이 더
단순하다(docs/superpowers/specs/2026-07-06-handover-ui-rclpy-design.md 참고).
"""

import json

from rclpy.node import Node
from std_msgs.msg import String

from handover_interfaces.msg import GripperState


class _HandoverUiNode(Node):
    """실제 ROS2 통신을 담당하는 rclpy 노드.

    구독 콜백은 파싱만 하고, 결과 전달은 owner(RosClient)의 콜백 속성
    (on_task_status 등)을 직접 호출한다. owner가 그 자리에 Qt pyqtSignal의
    emit을 꽂아두므로(main_window.py), 콜백이 spin 스레드에서 불려도 Qt가
    알아서 메인 스레드로 큐잉해 위젯을 안전하게 갱신한다.
    """

    def __init__(self, owner):
        super().__init__('handover_ui')
        self._owner = owner
        self._command_pub = None

    def subscribe_all(self):
        self.create_subscription(String, '/task/status', self._on_task_status, 10)
        self.create_subscription(GripperState, '/gripper/state', self._on_gripper_state, 10)
        self.create_subscription(String, '/robot/fault', self._on_fault, 10)
        self._command_pub = self.create_publisher(String, '/user_command/text', 10)

    def publish_command(self, text: str):
        self._command_pub.publish(String(data=text))

    def _on_task_status(self, msg):
        if self._owner.on_task_status is None:
            return
        payload = json.loads(msg.data)
        self._owner.on_task_status(payload.get('state', ''), payload.get('detail', ''))

    def _on_gripper_state(self, msg):
        if self._owner.on_gripper_state is None:
            return
        self._owner.on_gripper_state(msg.width_mm, msg.grip_detected)

    def _on_fault(self, msg):
        if self._owner.on_fault is None:
            return
        self._owner.on_fault(msg.data)
