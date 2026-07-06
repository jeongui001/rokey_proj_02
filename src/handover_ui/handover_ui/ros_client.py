"""rclpy로 ROS2에 직접 접속해 handover_ui가 필요한 토픽을 구독/퍼블리시한다.

원래는 rosbridge_server(WebSocket) + roslibpy로 접속했으나(브라우저 UI 초안의
흔적), UI가 PyQt 데스크톱 앱으로 확정된 뒤에는 rclpy로 직접 붙는 것이 더
단순하다(docs/superpowers/specs/2026-07-06-handover-ui-rclpy-design.md 참고).
"""

import json

import rclpy
from PyQt5.QtCore import QThread
from rclpy.executors import SingleThreadedExecutor
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


class RosSpinThread(QThread):
    """spin 루프만 도는 워커 스레드. 자체 시그널은 두지 않는다 - _HandoverUiNode가
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
    """handover_ui가 사용하는 통신 래퍼. MainWindow는 이 클래스의 콜백 슬롯
    (on_task_status/on_gripper_state/on_fault)에 자기 핸들러를 꽂아넣기만
    하면 되고, rclpy를 전혀 몰라도 된다(main_window.py 참고).
    """

    def __init__(self):
        self.on_task_status = None
        self.on_gripper_state = None
        self.on_fault = None
        self._node = _HandoverUiNode(self)
        self._spin_thread = RosSpinThread(self._node)

    def connect(self):
        self._spin_thread.start()

    def close(self):
        self._spin_thread.stop()
        self._node.destroy_node()

    def is_connected(self) -> bool:
        return self._spin_thread.isRunning()

    def subscribe_all(self):
        self._node.subscribe_all()

    def publish_command(self, text: str):
        self._node.publish_command(text)
