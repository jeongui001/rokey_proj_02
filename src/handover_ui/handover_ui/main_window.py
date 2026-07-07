from PyQt5 import QtWidgets, QtCore


class MainWindow(QtWidgets.QMainWindow):
    """공구 전달 로봇 제어 화면. RosClient가 백그라운드 스레드에서 받는 메시지를
    Qt 위젯 업데이트로 이어붙이는 게 이 클래스의 핵심 역할이다.

    RosClient의 콜백(on_task_status 등)은 rclpy 스핀 스레드(RosSpinThread)에서
    호출될 수 있는데, PyQt 위젯은 메인 스레드에서만 안전하게 갱신할 수 있다.
    그래서 콜백에서 위젯을 직접 만지지 않고 pyqtSignal을 emit만 하면, Qt가
    알아서 메인 스레드 큐에 넣어 _update_* 슬롯을 안전하게 실행해준다.
    """

    task_status_received = QtCore.pyqtSignal(str, str)
    gripper_state_received = QtCore.pyqtSignal(float, bool)
    fault_received = QtCore.pyqtSignal(str)

    def __init__(self, ros_client):
        super().__init__()
        self.ros_client = ros_client
        self.setWindowTitle('공구 전달 로봇 제어')

        self.state_label = QtWidgets.QLabel('상태: -')
        self.detail_label = QtWidgets.QLabel('디테일: -')
        self.gripper_label = QtWidgets.QLabel('그리퍼: -')
        self.fault_banner = QtWidgets.QLabel('')
        self.fault_banner.setStyleSheet('background-color: red; color: white; font-weight: bold;')
        self.fault_banner.hide()
        self.log_view = QtWidgets.QListWidget()

        self.command_input = QtWidgets.QLineEdit()
        self.send_button = QtWidgets.QPushButton('전송')
        self.send_button.clicked.connect(self._on_send_clicked)

        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(self.command_input)
        input_layout.addWidget(self.send_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.fault_banner)
        layout.addWidget(self.state_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.gripper_label)
        layout.addLayout(input_layout)
        layout.addWidget(self.log_view)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # signal -> slot 연결 (Qt 메인 스레드에서 안전하게 실행되도록)
        self.task_status_received.connect(self._update_task_status)
        self.gripper_state_received.connect(self._update_gripper_state)
        self.fault_received.connect(self._update_fault)

        # ros_client의 콜백 슬롯에 "emit"을 꽂아넣는다 - 메시지가 오면
        # 위젯을 직접 바꾸는 대신 신호만 쏘게 하기 위함(클래스 docstring 참고)
        self.ros_client.on_task_status = self.task_status_received.emit
        self.ros_client.on_gripper_state = self.gripper_state_received.emit
        self.ros_client.on_fault = self.fault_received.emit

    def _on_send_clicked(self):
        text = self.command_input.text().strip()
        if not text:
            return
        self.ros_client.publish_command(text)
        self.command_input.clear()

    def _update_task_status(self, state, detail):
        self.state_label.setText(f'상태: {state}')
        self.detail_label.setText(f'디테일: {detail}')
        self.log_view.addItem(f'[{state}] {detail}')

    def _update_gripper_state(self, width_mm, grip_detected):
        self.gripper_label.setText(f'그리퍼: {width_mm:.1f}mm, grip_detected={grip_detected}')

    def _update_fault(self, message):
        self.fault_banner.setText(f'FAULT: {message}')
        self.fault_banner.show()
        self.log_view.addItem(f'[FAULT] {message}')
