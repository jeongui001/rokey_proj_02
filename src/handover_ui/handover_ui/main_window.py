from PyQt5 import QtWidgets, QtCore


class MainWindow(QtWidgets.QMainWindow):
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

        self.task_status_received.connect(self._update_task_status)
        self.gripper_state_received.connect(self._update_gripper_state)
        self.fault_received.connect(self._update_fault)

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
