import time

from PyQt5 import QtCore, QtGui, QtWidgets

# safety_state -> (배경색, 글자색). NORMAL은 초록, PROTECTIVE_STOP/RECOVERY_REQUIRED는
# 주황, EMERGENCY_STOP/FAULT는 빨강. 정의되지 않은 값은 회색(기본색)으로 표시한다.
SAFETY_STATE_COLORS = {
    'NORMAL': ('#2e7d32', 'white'),
    'PROTECTIVE_STOP': ('#e65100', 'white'),
    'RECOVERY_REQUIRED': ('#e65100', 'white'),
    'EMERGENCY_STOP': ('#c62828', 'white'),
    'FAULT': ('#c62828', 'white'),
}
DEFAULT_SAFETY_COLOR = ('#757575', 'white')


def _severity_from_fault_message(message: str) -> str:
    """/robot/fault 메시지의 접두어로 심각도를 판정한다.

    PROTECTIVE_STOP -> 주황, EMERGENCY_STOP/FAULT -> 빨강, 접두어가 없으면
    안전을 위해 빨강(FAULT)으로 간주한다 (요구사항: 접두어 없음 -> 빨강).
    """
    text = message or ''
    if text.startswith('PROTECTIVE_STOP'):
        return 'PROTECTIVE_STOP'
    if text.startswith('EMERGENCY_STOP'):
        return 'EMERGENCY_STOP'
    if text.startswith('FAULT'):
        return 'FAULT'
    return 'FAULT'

# 버튼 -> /user_command/text로 보낼 정확한 문구 (task_manager의 command_parser와 합의됨).
CMD_AUTO = '자동 모드로 전환해'
CMD_MANUAL = '수동 모드로 전환해'
CMD_HOME = '홈으로 가'
CMD_FRONT = '정면을 봐'
CMD_UP = '위를 봐'
CMD_DOWN = '아래를 봐'
CMD_WATCH = '컨베이어를 봐'
CMD_STOP = '멈춰'
CMD_RESET = '리셋'


class MainWindow(QtWidgets.QMainWindow):
    task_status_received = QtCore.pyqtSignal(str, str, str, str)  # state, detail, operation_mode, safety_state
    gripper_state_received = QtCore.pyqtSignal(float, bool)
    fault_received = QtCore.pyqtSignal(str)
    connection_changed = QtCore.pyqtSignal(bool)
    camera_image_received = QtCore.pyqtSignal(bytes)

    def __init__(self, ros_client):
        super().__init__()
        self.ros_client = ros_client
        self.setWindowTitle('공구 전달 로봇 제어')

        self._last_state = None
        self._last_operation_mode = None
        self._last_safety_state = None
        self._last_grip_detected = None
        self._last_fault_message = None
        # /task/status가 fault 이후 실제로 비정상 safety_state를 확인해줘야만
        # 그다음 NORMAL이 배너를 숨길 수 있다 (fault 직후 도착하는 과거의 NORMAL
        # 메시지가 배너를 바로 숨겨버리지 않도록 하는 안전장치).
        self._fault_confirmed_abnormal = False
        self._camera_image = None

        self._build_ui()
        self._wire_signals()

        self._reconnect_timer = QtCore.QTimer(self)
        self._reconnect_timer.setInterval(int(self.ros_client.reconnect_interval_s * 1000))
        self._reconnect_timer.timeout.connect(self.ros_client.ensure_connected)
        self._reconnect_timer.start()

    # ---- UI 구성 ----

    def _build_ui(self):
        # 상단: rosbridge 연결 상태 / AUTO·MANUAL / task state / safety state / detail
        self.connection_label = QtWidgets.QLabel('rosbridge: 연결 확인 중...')
        self.mode_label = QtWidgets.QLabel('모드: -')
        self.state_label = QtWidgets.QLabel('상태: -')
        self.safety_label = QtWidgets.QLabel('안전상태: -')
        self.detail_label = QtWidgets.QLabel('디테일: -')
        for label in (self.connection_label, self.mode_label, self.state_label,
                      self.safety_label, self.detail_label):
            label.setStyleSheet('padding: 2px 6px;')

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self.connection_label)
        top_bar.addWidget(self.mode_label)
        top_bar.addWidget(self.state_label)
        top_bar.addWidget(self.safety_label)
        top_bar.addWidget(self.detail_label, stretch=1)

        self.fault_banner = QtWidgets.QLabel('')
        self.fault_banner.setStyleSheet('background-color: #c62828; color: white; font-weight: bold; padding: 4px;')
        self.fault_banner.hide()

        # 왼쪽: 카메라
        self.camera_label = QtWidgets.QLabel('카메라 대기 중...')
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet('background-color: #202020; color: #aaaaaa;')

        # 오른쪽: 그리퍼 상태 + 명령 버튼
        self.gripper_label = QtWidgets.QLabel('그리퍼: -')

        self.auto_button = QtWidgets.QPushButton('AUTO 모드')
        self.manual_button = QtWidgets.QPushButton('MANUAL 모드')
        self.auto_button.clicked.connect(lambda: self._send_text(CMD_AUTO))
        self.manual_button.clicked.connect(lambda: self._send_text(CMD_MANUAL))
        mode_button_layout = QtWidgets.QHBoxLayout()
        mode_button_layout.addWidget(self.auto_button)
        mode_button_layout.addWidget(self.manual_button)

        self.home_button = QtWidgets.QPushButton('홈')
        self.front_button = QtWidgets.QPushButton('정면')
        self.up_button = QtWidgets.QPushButton('위')
        self.down_button = QtWidgets.QPushButton('아래')
        self.watch_button = QtWidgets.QPushButton('컨베이어')
        self.home_button.clicked.connect(lambda: self._send_text(CMD_HOME))
        self.front_button.clicked.connect(lambda: self._send_text(CMD_FRONT))
        self.up_button.clicked.connect(lambda: self._send_text(CMD_UP))
        self.down_button.clicked.connect(lambda: self._send_text(CMD_DOWN))
        self.watch_button.clicked.connect(lambda: self._send_text(CMD_WATCH))
        pose_grid = QtWidgets.QGridLayout()
        pose_grid.addWidget(self.home_button, 0, 0)
        pose_grid.addWidget(self.front_button, 0, 1)
        pose_grid.addWidget(self.watch_button, 0, 2)
        pose_grid.addWidget(self.up_button, 1, 0)
        pose_grid.addWidget(self.down_button, 1, 1)

        self.stop_button = QtWidgets.QPushButton('작업 중단')
        self.stop_button.clicked.connect(lambda: self._send_text(CMD_STOP))
        self.estop_notice_label = QtWidgets.QLabel(
            '※ 실제 비상정지(E-Stop)는 로봇 본체의 물리 버튼입니다.')
        self.estop_notice_label.setStyleSheet('color: #757575; font-size: 11px;')
        self.estop_notice_label.setWordWrap(True)

        self.reset_button = QtWidgets.QPushButton('복구 요청 (리셋)')
        self.reset_button.clicked.connect(lambda: self._send_text(CMD_RESET))

        self.command_input = QtWidgets.QLineEdit()
        self.send_button = QtWidgets.QPushButton('전송')
        self.send_button.clicked.connect(self._on_send_clicked)
        free_command_layout = QtWidgets.QHBoxLayout()
        free_command_layout.addWidget(self.command_input)
        free_command_layout.addWidget(self.send_button)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(self.gripper_label)
        right_panel.addLayout(mode_button_layout)
        right_panel.addLayout(pose_grid)
        right_panel.addWidget(self.stop_button)
        right_panel.addWidget(self.estop_notice_label)
        right_panel.addWidget(self.reset_button)
        right_panel.addStretch(1)
        right_panel.addLayout(free_command_layout)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        right_container.setMaximumWidth(260)

        middle_layout = QtWidgets.QHBoxLayout()
        middle_layout.addWidget(self.camera_label, stretch=2)
        middle_layout.addWidget(right_container, stretch=1)

        # 하단: 시간순 로그
        self.log_view = QtWidgets.QListWidget()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.fault_banner)
        layout.addLayout(middle_layout, stretch=1)
        layout.addWidget(self.log_view, stretch=1)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(900, 640)

    def _wire_signals(self):
        self.task_status_received.connect(self._update_task_status)
        self.gripper_state_received.connect(self._update_gripper_state)
        self.fault_received.connect(self._update_fault)
        self.connection_changed.connect(self._update_connection_status)
        self.camera_image_received.connect(self._update_camera_image)

        self.ros_client.on_task_status = self.task_status_received.emit
        self.ros_client.on_gripper_state = self.gripper_state_received.emit
        self.ros_client.on_fault = self.fault_received.emit
        self.ros_client.on_connection_changed = self.connection_changed.emit
        self.ros_client.on_camera_image = self.camera_image_received.emit

    # ---- 명령 전송 ----

    def _on_send_clicked(self):
        text = self.command_input.text().strip()
        if not text:
            return
        self._send_text(text)
        self.command_input.clear()

    def _send_text(self, text):
        text = (text or '').strip()
        if not text:
            return
        try:
            sent = self.ros_client.publish_command(text)
        except Exception as exc:  # 전송 중 예외로 UI가 죽지 않도록 방어
            self._log(f'명령 전송 중 오류: {text} ({exc})')
            return
        if sent:
            self._log(f'명령 전송: {text}')
        else:
            reason = 'rosbridge 미연결' if not self.ros_client.is_connected() else '빈 명령'
            self._log(f'명령 전송 실패({reason}): {text}')

    # ---- 상태 업데이트 ----

    def _update_task_status(self, state, detail, operation_mode, safety_state):
        self.state_label.setText(f'상태: {state}')
        self.detail_label.setText(f'디테일: {detail}')
        self.mode_label.setText(f'모드: {operation_mode or "-"}')

        if self._last_state is not None and state != self._last_state:
            self._log(f'작업 상태 변화: {self._last_state} -> {state}')
        self._last_state = state

        if self._last_operation_mode is not None and operation_mode != self._last_operation_mode:
            self._log(f'모드 전환: {self._last_operation_mode} -> {operation_mode}')
        self._last_operation_mode = operation_mode

        self._apply_safety_state(safety_state)

    def _apply_safety_state(self, safety_state):
        if safety_state != self._last_safety_state:
            self._log(f'안전상태 변화: {self._last_safety_state or "-"} -> {safety_state}')
        self._last_safety_state = safety_state

        bg, fg = SAFETY_STATE_COLORS.get(safety_state, DEFAULT_SAFETY_COLOR)
        self.safety_label.setText(f'안전상태: {safety_state or "-"}')
        self.safety_label.setStyleSheet(f'background-color: {bg}; color: {fg}; padding: 2px 6px;')

        if safety_state != 'NORMAL':
            # /task/status 스스로 비정상 상태를 확인해줬다 - 이후 NORMAL만 배너를 숨길 수 있다.
            self._fault_confirmed_abnormal = True
            if not self.fault_banner.isVisible():
                text = self._last_fault_message or f'안전상태: {safety_state}'
                self.fault_banner.setText(text)
                self.fault_banner.setStyleSheet(
                    f'background-color: {bg}; color: {fg}; font-weight: bold; padding: 4px;')
                self.fault_banner.show()
            return

        # safety_state == 'NORMAL': 이전에 /task/status가 비정상 상태를 실제로
        # 확인해준 적이 있을 때만(=진짜 복구 확인) 배너를 숨긴다. fault 메시지 직후
        # 도착한, 아직 갱신되지 않은 과거의 NORMAL 메시지는 무시하고 배너를 유지한다.
        if self._fault_confirmed_abnormal:
            self._last_fault_message = None
            self._fault_confirmed_abnormal = False
            self.fault_banner.hide()

    def _update_gripper_state(self, width_mm, grip_detected):
        self.gripper_label.setText(f'그리퍼: {width_mm:.1f}mm, grip_detected={grip_detected}')
        if self._last_grip_detected is not None and grip_detected != self._last_grip_detected:
            self._log(f'그리퍼 상태 변화: grip_detected={grip_detected}')
        self._last_grip_detected = grip_detected

    def _update_fault(self, message):
        self._last_fault_message = message
        # 배너 색은 이전 safety_state가 아니라 이 메시지 자체의 접두어로 정한다 -
        # 그래야 직전 safety_state가 NORMAL이었어도 초록으로 표시되는 일이 없다.
        severity = _severity_from_fault_message(message)
        # UI 내부 안전상태도 즉시 비정상으로 간주한다 (/task/status 갱신을 기다리지 않음).
        self._last_safety_state = severity
        bg, fg = SAFETY_STATE_COLORS.get(severity, ('#c62828', 'white'))
        self.safety_label.setText(f'안전상태: {severity}')
        self.safety_label.setStyleSheet(f'background-color: {bg}; color: {fg}; padding: 2px 6px;')

        self.fault_banner.setText(message)
        self.fault_banner.setStyleSheet(
            f'background-color: {bg}; color: {fg}; font-weight: bold; padding: 4px;')
        self.fault_banner.show()
        self._log(f'Fault: {message}')

    def _update_connection_status(self, connected):
        if connected:
            self.connection_label.setText('rosbridge: 연결됨')
            self.connection_label.setStyleSheet('color: white; background-color: #2e7d32; padding: 2px 6px;')
        else:
            self.connection_label.setText('rosbridge: 연결 안 됨')
            self.connection_label.setStyleSheet('color: white; background-color: #c62828; padding: 2px 6px;')
        self._log('rosbridge 연결됨' if connected else 'rosbridge 연결 끊김')

    # ---- 카메라 ----

    def _update_camera_image(self, image_bytes):
        image = QtGui.QImage.fromData(image_bytes)
        if image.isNull():
            return
        self._camera_image = image
        self._render_camera_pixmap()

    def _render_camera_pixmap(self):
        if self._camera_image is None:
            return
        pixmap = QtGui.QPixmap.fromImage(self._camera_image)
        scaled = pixmap.scaled(
            self.camera_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_camera_pixmap()

    # ---- 로그 ----

    def _log(self, message):
        timestamp = time.strftime('%H:%M:%S')
        self.log_view.addItem(f'[{timestamp}] {message}')
        self.log_view.scrollToBottom()

    # ---- 종료 ----

    def closeEvent(self, event):
        self._reconnect_timer.stop()
        try:
            self.ros_client.close()
        except Exception:
            pass
        super().closeEvent(event)
