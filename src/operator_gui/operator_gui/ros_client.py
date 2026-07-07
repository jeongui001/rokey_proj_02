"""rclpy로 ROS2에 직접 접속해 operator_gui가 필요한 토픽을 구독/퍼블리시한다.

원래는 rosbridge_server(WebSocket) + roslibpy로 접속했으나(브라우저 UI 초안의
흔적), UI가 PyQt 데스크톱 앱으로 확정된 뒤에는 rclpy로 직접 붙는 것이 더
단순하다. rosbridge와 달리 연결 성립이 비동기 핸드셰이크가 아니라 노드 생성 +
구독 등록만으로 즉시 확정되므로, roslibpy 시절의 재연결/재구독 로직은 필요 없다 -
남아있는 유일한 실패 모드는 spin 스레드 자체가 죽는 경우뿐이라 ensure_connected()는
그것만 확인한다.
"""

import json
import os

import rclpy
from PyQt5.QtCore import QThread
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from handover_interfaces.msg import GripperState

DEFAULT_CAMERA_TOPIC = '/vision/debug_image/compressed'
DEFAULT_RECONNECT_INTERVAL_S = 5.0


class _OperatorGuiNode(Node):
    """실제 ROS2 통신을 담당하는 rclpy 노드.

    구독 콜백은 파싱만 하고, 결과 전달은 owner(RosClient)의 콜백 속성
    (on_task_status 등)을 직접 호출한다. owner가 그 자리에 Qt pyqtSignal의
    emit을 꽂아두므로(main_window.py), 콜백이 spin 스레드에서 불려도 Qt가
    알아서 메인 스레드로 큐잉해 위젯을 안전하게 갱신한다.
    """

    def __init__(self, owner, camera_topic):
        super().__init__('operator_gui')
        self._owner = owner
        self._camera_topic = camera_topic
        self._command_pub = None

    def subscribe_all(self):
        self.create_subscription(String, '/task/status', self._on_task_status, 10)
        self.create_subscription(GripperState, '/gripper/state', self._on_gripper_state, 10)
        self.create_subscription(String, '/robot/fault', self._on_fault, 10)
        self.create_subscription(
            CompressedImage, self._camera_topic, self._on_camera_image, 10)
        self._command_pub = self.create_publisher(String, '/user_command/text', 10)

    def publish_command(self, text: str) -> bool:
        text = (text or '').strip()
        if not text or self._command_pub is None:
            return False
        self._command_pub.publish(String(data=text))
        return True

    def _on_task_status(self, msg):
        if self._owner.on_task_status is None:
            return
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self._owner.on_task_status(
            payload.get('state', ''), payload.get('detail', ''),
            payload.get('operation_mode', ''), payload.get('safety_state', ''))

    def _on_gripper_state(self, msg):
        if self._owner.on_gripper_state is None:
            return
        self._owner.on_gripper_state(msg.width_mm, msg.grip_detected)

    def _on_fault(self, msg):
        if self._owner.on_fault is None:
            return
        self._owner.on_fault(msg.data)

    def _on_camera_image(self, msg):
        if self._owner.on_camera_image is None:
            return
        self._owner.on_camera_image(bytes(msg.data))


class RosSpinThread(QThread):
    """spin 루프만 도는 워커 스레드. 자체 시그널은 두지 않는다 - _OperatorGuiNode가
    owner 콜백을 직접 호출하고, pyqtSignal.emit()은 호출 스레드와 무관하게
    수신측 스레드로 큐잉되므로 별도 중계가 필요 없다."""

    def __init__(self, node):
        super().__init__()
        self.node = node
        # 노드 전용 executor를 직접 들고 있는다 - rclpy.spin_once(node)처럼
        # 인자 없이 쓰면 프로세스 전역 executor(get_global_executor())를
        # 공유하는데, 이 스레드가 매 0.1s 그 전역 executor를 계속 점유하는
        # 동안 다른 스레드가 다른 노드로 rclpy.spin_once를 호출하면
        # "generator already executing" 예외가 난다(전역 executor가 동시에
        # 두 번 진입됨). 노드 전용 executor를 쓰면 이 스레드는 완전히
        # 격리되어 다른 스레드의 spin과 절대 부딪히지 않는다.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._running = False

    def start(self):
        # 메인 스레드에서 미리 True로 세팅해야 한다 - run() 안에서 세팅하면
        # "start() 직후 바로 stop()"처럼 빠르게 연달아 호출될 때 백그라운드
        # 스레드가 아직 run()에 진입하지 못한 사이 stop()이 _running=False로
        # 바꿔놔도 뒤늦게 run()이 그걸 다시 True로 덮어써 루프가 영원히 도는
        # 경쟁 상태(race)가 생긴다.
        self._running = True
        super().start()

    def run(self):
        while self._running and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def stop(self):
        self._running = False
        self.wait()
        self._executor.shutdown()


class RosClient:
    """operator_gui가 사용하는 통신 래퍼. MainWindow는 이 클래스의 콜백 슬롯
    (on_task_status/on_gripper_state/on_fault/on_connection_changed/
    on_camera_image)에 자기 핸들러를 꽂아넣기만 하면 되고, rclpy를 전혀
    몰라도 된다(main_window.py 참고).
    """

    def __init__(self, camera_topic=None, reconnect_interval_s=DEFAULT_RECONNECT_INTERVAL_S):
        self.camera_topic = camera_topic or os.environ.get(
            'OPERATOR_GUI_CAMERA_TOPIC', DEFAULT_CAMERA_TOPIC)
        self.reconnect_interval_s = reconnect_interval_s

        self.on_task_status = None        # (state, detail, operation_mode, safety_state)
        self.on_gripper_state = None       # (width_mm, grip_detected)
        self.on_fault = None               # (message)
        self.on_camera_image = None        # (image_bytes)
        self.on_connection_changed = None  # (is_connected: bool)

        self._node = _OperatorGuiNode(self, self.camera_topic)
        self._spin_thread = RosSpinThread(self._node)
        self._closed = False

    def connect(self):
        self._closed = False
        if not self._spin_thread.isRunning():
            self._spin_thread.start()
        if self.on_connection_changed is not None:
            self.on_connection_changed(True)

    def ensure_connected(self):
        """spin 스레드가 살아있는지 확인한다. UI 쪽 QTimer가 주기적으로 호출한다 -
        rclpy는 노드 생성 시점에 연결이 즉시 확정되므로, 남은 유일한 실패 모드인
        spin 스레드 생존만 확인하면 된다(구독 자체는 스레드 상태와 무관하게
        노드에 남아있으므로 재구독은 필요 없다)."""
        if self._closed:
            return
        if self.is_connected():
            return
        if self.on_connection_changed is not None:
            self.on_connection_changed(False)
        self.connect()

    def close(self):
        self._closed = True
        self._spin_thread.stop()
        self._node.destroy_node()

    def is_connected(self) -> bool:
        return self._spin_thread.isRunning()

    def subscribe_all(self):
        self._node.subscribe_all()

    def publish_command(self, text: str) -> bool:
        return self._node.publish_command(text)
