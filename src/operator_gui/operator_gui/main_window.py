import math
import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets

# 카메라 프레임이 이 시간(초) 이상 없으면 "멈췄습니다"로 표시한다. 실제 카메라
# 프레임 주기에 맞춰 조정 가능한 통신 타이밍 값이며, 하드웨어 확정값이 아니다.
DEFAULT_CAMERA_STALE_TIMEOUT_S = 2.0
# 위 timeout과 무관하게, 화면 갱신 자체를 확인하는 주기(ms). timeout보다 충분히
# 짧게 유지해 멈춤 표시가 늦게 뜨지 않게 한다.
_CAMERA_STALE_CHECK_INTERVAL_MS = 300

_MONO_FONT_STACK = '"Cascadia Code", "DejaVu Sans Mono", "Consolas", monospace'


def _sanitize_camera_stale_timeout_s(value):
    """finite 양수만 허용한다.

    환경변수/생성자 인자로 들어온 값이 파싱 불가능한 문자열이거나 NaN/Inf/0
    이하이면 안전한 기본값(DEFAULT_CAMERA_STALE_TIMEOUT_S)으로 대체한다 - 잘못된
    값 하나 때문에 GUI가 시작 중 죽거나 즉시 "멈췄습니다"만 표시하는 등 비정상
    동작하지 않게 한다."""
    if value is None:
        return DEFAULT_CAMERA_STALE_TIMEOUT_S
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_CAMERA_STALE_TIMEOUT_S
    if not math.isfinite(parsed) or parsed <= 0.0:
        return DEFAULT_CAMERA_STALE_TIMEOUT_S
    return parsed

# safety_state -> (glow 색, 은은한 배경 tint). NORMAL은 초록, PROTECTIVE_STOP/
# RECOVERY_REQUIRED는 주황, EMERGENCY_STOP/FAULT는 빨강 - HUD 톤에 맞춰 어두운
# 배경 위에서 빛나는 느낌으로 표현한다. 정의되지 않은 값은 회색 계열로 표시한다.
SAFETY_STATE_COLORS = {
    'NORMAL': ('#39e991', 'rgba(57, 233, 145, 0.14)'),
    'PROTECTIVE_STOP': ('#ffb454', 'rgba(255, 180, 84, 0.16)'),
    'RECOVERY_REQUIRED': ('#ffb454', 'rgba(255, 180, 84, 0.16)'),
    'EMERGENCY_STOP': ('#ff4d5e', 'rgba(255, 77, 94, 0.18)'),
    'FAULT': ('#ff4d5e', 'rgba(255, 77, 94, 0.18)'),
}
DEFAULT_SAFETY_COLOR = ('#7f95a3', 'rgba(127, 149, 163, 0.12)')

# 앱 전역 다크 HUD 테마. safety_label/fault_banner/connection_label처럼 상태에
# 따라 색이 바뀌는 위젯은 이 QSS와 별개로 각자 setStyleSheet()을 계속 사용한다
# (이 전역 QSS는 그 외 정적인 부분 - 배경, 버튼, 그룹박스, 입력창, 로그 - 을 담당).
_JARVIS_QSS = f"""
QMainWindow {{
    background-color: #05080d;
}}
QWidget {{
    color: #cfe6f2;
    background-color: transparent;
}}
QGroupBox {{
    border: 1px solid #16324a;
    border-radius: 3px;
    margin-top: 16px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: #cfe6f2;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    padding: 0 6px;
    color: #5f7c8e;
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton {{
    background-color: #0f1c28;
    color: #cfe6f2;
    border: 1px solid #2c5a7d;
    border-radius: 4px;
    padding: 9px 12px;
    font-size: 13px;
}}
QPushButton:hover {{
    border-color: #4dd8ff;
    color: #4dd8ff;
}}
QPushButton:pressed {{
    background-color: #16324a;
}}
QPushButton:disabled {{
    color: #3d4f5c;
    border-color: #1c2a33;
    background-color: #0a1119;
}}
QPushButton#primaryButton {{
    background-color: #4dd8ff;
    color: #05141c;
    border: 1px solid #4dd8ff;
    font-weight: 700;
}}
QPushButton#primaryButton:hover {{
    background-color: #7ee4ff;
    color: #05141c;
}}
QPushButton#primaryButton:disabled {{
    background-color: #123544;
    color: #3d4f5c;
    border-color: #1c2a33;
}}
QPushButton#dangerButton {{
    background-color: transparent;
    color: #ff4d5e;
    border: 1px solid #ff4d5e;
    font-weight: 700;
}}
QPushButton#dangerButton:hover {{
    background-color: #ff4d5e;
    color: #05080d;
}}
QLineEdit {{
    background-color: #0c141d;
    color: #cfe6f2;
    border: 1px solid #2c5a7d;
    border-radius: 4px;
    padding: 7px 9px;
}}
QLineEdit:focus {{
    border-color: #4dd8ff;
}}
QListWidget {{
    background-color: #0c141d;
    color: #8fb3c4;
    border: 1px solid #16324a;
    border-radius: 3px;
    font-family: {_MONO_FONT_STACK};
    font-size: 12px;
}}
QLabel#sectionNote {{
    color: #5f7c8e;
    font-size: 11px;
}}
QLabel#cameraLabel {{
    background-color: #0a0f16;
    border: 1px solid #16324a;
    border-radius: 3px;
    color: #4d6577;
    font-family: {_MONO_FONT_STACK};
}}
QScrollBar:vertical {{
    background: #0c141d;
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: #2c5a7d;
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


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

    def __init__(self, ros_client, camera_stale_timeout_s=None):
        super().__init__()
        self.ros_client = ros_client
        self.setWindowTitle('공구 전달 로봇 제어')

        self.camera_stale_timeout_s = _sanitize_camera_stale_timeout_s(
            camera_stale_timeout_s if camera_stale_timeout_s is not None
            else os.environ.get('OPERATOR_GUI_CAMERA_STALE_TIMEOUT_S'))

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
        # 마지막으로 정상 영상을 받은 시각(monotonic). None이면 아직 한 번도 받지
        # 못한 것 - 이때는 "카메라 대기 중..."을 그대로 유지한다(멈춤으로 표시하지 않음).
        self._last_camera_image_at = None
        self._camera_stale = False
        # 초기 /task/status를 받기 전에는 이동/모드 버튼을 안전하게 비활성화해 둔다.
        self._received_first_status = False

        self._build_ui()
        self._wire_signals()
        self._apply_button_policy()

        self._reconnect_timer = QtCore.QTimer(self)
        self._reconnect_timer.setInterval(int(self.ros_client.reconnect_interval_s * 1000))
        self._reconnect_timer.timeout.connect(self.ros_client.ensure_connected)
        self._reconnect_timer.start()

        self._camera_stale_timer = QtCore.QTimer(self)
        self._camera_stale_timer.setInterval(_CAMERA_STALE_CHECK_INTERVAL_MS)
        self._camera_stale_timer.timeout.connect(self._check_camera_stale)
        self._camera_stale_timer.start()

    # ---- UI 구성 ----

    def _build_ui(self):
        self.setStyleSheet(_JARVIS_QSS)

        # 상단: ROS2 연결 상태 / AUTO·MANUAL / task state / safety state / detail
        self.connection_label = QtWidgets.QLabel('ROS2: 연결 확인 중...')
        self.mode_label = QtWidgets.QLabel('모드: -')
        self.state_label = QtWidgets.QLabel('상태: -')
        self.safety_label = QtWidgets.QLabel('안전상태: -')
        self.detail_label = QtWidgets.QLabel('디테일: -')

        _chip_base = (
            f'font-family: {_MONO_FONT_STACK}; font-size: 12px; font-weight: 600; '
            'padding: 6px 12px; border: 1px solid #16324a; border-radius: 3px; '
            'background-color: #0c141d; color: #8fb3c4;'
        )
        for label in (self.connection_label, self.mode_label, self.state_label):
            label.setStyleSheet(_chip_base)
        self.detail_label.setStyleSheet(
            'font-size: 12px; color: #5f7c8e; padding: 6px 4px;')
        self.safety_label.setStyleSheet(_chip_base)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(8)
        top_bar.addWidget(self.connection_label)
        top_bar.addWidget(self.mode_label)
        top_bar.addWidget(self.state_label)
        top_bar.addWidget(self.safety_label)
        top_bar.addWidget(self.detail_label, stretch=1)

        self.fault_banner = QtWidgets.QLabel('')
        self.fault_banner.setStyleSheet(
            f'background-color: rgba(255, 77, 94, 0.18); color: #ff4d5e; '
            f'font-family: {_MONO_FONT_STACK}; font-weight: 700; '
            'border: 1px solid #ff4d5e; border-radius: 3px; padding: 6px 10px;')
        self.fault_banner.hide()

        # 안전상태 배너가 위험할 때 은은하게 깜빡이도록(pulse) 하는 애니메이션 -
        # NORMAL일 때는 항상 정지 상태(불투명도 1.0)로 둔다.
        self._safety_opacity_effect = QtWidgets.QGraphicsOpacityEffect(self.safety_label)
        self._safety_opacity_effect.setOpacity(1.0)
        self.safety_label.setGraphicsEffect(self._safety_opacity_effect)
        fade_out = QtCore.QPropertyAnimation(self._safety_opacity_effect, b'opacity', self)
        fade_out.setDuration(700)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.45)
        fade_in = QtCore.QPropertyAnimation(self._safety_opacity_effect, b'opacity', self)
        fade_in.setDuration(700)
        fade_in.setStartValue(0.45)
        fade_in.setEndValue(1.0)
        self._safety_pulse = QtCore.QSequentialAnimationGroup(self)
        self._safety_pulse.addAnimation(fade_out)
        self._safety_pulse.addAnimation(fade_in)
        self._safety_pulse.setLoopCount(-1)

        # 왼쪽: 카메라
        self.camera_label = QtWidgets.QLabel('카메라 대기 중...')
        self.camera_label.setObjectName('cameraLabel')
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)

        # 오른쪽: 그리퍼 상태 + 명령 버튼 (모드 / 이동 / 비상 조치 / 명령 전송으로 그룹화)
        self.gripper_label = QtWidgets.QLabel('그리퍼: -')
        self.gripper_label.setStyleSheet(f'font-family: {_MONO_FONT_STACK}; font-size: 12px;')

        mode_group = QtWidgets.QGroupBox('모드')
        self.auto_button = QtWidgets.QPushButton('AUTO 모드')
        self.manual_button = QtWidgets.QPushButton('MANUAL 모드')
        self.auto_button.clicked.connect(lambda: self._send_text(CMD_AUTO))
        self.manual_button.clicked.connect(lambda: self._send_text(CMD_MANUAL))
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(self.auto_button)
        mode_layout.addWidget(self.manual_button)
        mode_group.setLayout(mode_layout)

        move_group = QtWidgets.QGroupBox('이동')
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
        move_group.setLayout(pose_grid)

        estop_group = QtWidgets.QGroupBox('비상 조치')
        self.stop_button = QtWidgets.QPushButton('작업 중단')
        self.stop_button.setObjectName('dangerButton')
        self.stop_button.clicked.connect(lambda: self._send_text(CMD_STOP))
        self.reset_button = QtWidgets.QPushButton('복구 요청 (리셋)')
        self.reset_button.setObjectName('primaryButton')
        self.reset_button.clicked.connect(lambda: self._send_text(CMD_RESET))
        self.estop_notice_label = QtWidgets.QLabel(
            '※ 실제 비상정지(E-Stop)는 로봇 본체의 물리 버튼입니다.')
        self.estop_notice_label.setObjectName('sectionNote')
        self.estop_notice_label.setWordWrap(True)
        estop_layout = QtWidgets.QVBoxLayout()
        estop_layout.addWidget(self.stop_button)
        estop_layout.addWidget(self.reset_button)
        estop_layout.addWidget(self.estop_notice_label)
        estop_group.setLayout(estop_layout)

        cmd_group = QtWidgets.QGroupBox('명령 전송')
        self.command_input = QtWidgets.QLineEdit()
        self.command_input.setPlaceholderText('예: 드라이버 가져와')
        self.send_button = QtWidgets.QPushButton('전송')
        self.send_button.clicked.connect(self._on_send_clicked)
        free_command_layout = QtWidgets.QHBoxLayout()
        free_command_layout.addWidget(self.command_input)
        free_command_layout.addWidget(self.send_button)
        cmd_group.setLayout(free_command_layout)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(10)
        right_panel.addWidget(self.gripper_label)
        right_panel.addWidget(mode_group)
        right_panel.addWidget(move_group)
        right_panel.addWidget(estop_group)
        right_panel.addStretch(1)
        right_panel.addWidget(cmd_group)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        right_container.setMaximumWidth(280)

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
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(960, 680)

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
            reason = 'ROS2 미연결' if not self.ros_client.is_connected() else '빈 명령'
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
        self._received_first_status = True
        self._apply_button_policy()

    def _style_safety_chip(self, widget, glow, tint):
        widget.setStyleSheet(
            f'font-family: {_MONO_FONT_STACK}; font-size: 12px; font-weight: 700; '
            f'padding: 6px 12px; border: 1px solid {glow}; border-radius: 3px; '
            f'background-color: {tint}; color: {glow};')

    def _apply_safety_pulse(self, safety_state):
        if safety_state == 'NORMAL':
            self._safety_pulse.stop()
            self._safety_opacity_effect.setOpacity(1.0)
        elif self._safety_pulse.state() != QtCore.QAbstractAnimation.Running:
            self._safety_pulse.start()

    def _apply_safety_state(self, safety_state):
        if safety_state != self._last_safety_state:
            self._log(f'안전상태 변화: {self._last_safety_state or "-"} -> {safety_state}')
        self._last_safety_state = safety_state

        glow, tint = SAFETY_STATE_COLORS.get(safety_state, DEFAULT_SAFETY_COLOR)
        self.safety_label.setText(f'안전상태: {safety_state or "-"}')
        self._style_safety_chip(self.safety_label, glow, tint)
        self._apply_safety_pulse(safety_state)

        if safety_state != 'NORMAL':
            # /task/status 스스로 비정상 상태를 확인해줬다 - 이후 NORMAL만 배너를 숨길 수 있다.
            self._fault_confirmed_abnormal = True
            if not self.fault_banner.isVisible():
                text = self._last_fault_message or f'안전상태: {safety_state}'
                self.fault_banner.setText(text)
                self.fault_banner.setStyleSheet(
                    f'background-color: {tint}; color: {glow}; '
                    f'font-family: {_MONO_FONT_STACK}; font-weight: 700; '
                    f'border: 1px solid {glow}; border-radius: 3px; padding: 6px 10px;')
                self.fault_banner.show()
            return

        # safety_state == 'NORMAL': 이전에 /task/status가 비정상 상태를 실제로
        # 확인해준 적이 있을 때만(=진짜 복구 확인) 배너를 숨긴다. fault 메시지 직후
        # 도착한, 아직 갱신되지 않은 과거의 NORMAL 메시지는 무시하고 배너를 유지한다.
        if self._fault_confirmed_abnormal:
            self._last_fault_message = None
            self._fault_confirmed_abnormal = False
            self.fault_banner.hide()

    # ---- 버튼 활성화 정책 (UI 편의 기능 - 실제 안전 판단은 task_manager/robot_control 담당) ----

    def _apply_button_policy(self):
        """현재 상태에 맞게 이동/모드 버튼을 활성화·비활성화한다.

        이 정책은 화면 편의 기능일 뿐이다 - 실제로 어떤 명령을 받아들일지는
        task_manager/robot_control이 최종 판단한다(예: MANUAL이 아닐 때 pose 이동
        명령을 보내도 task_manager가 거부한다). 여기서는 사용자가 애초에 거부될
        명령을 누르지 않도록 안내할 뿐이다.

        - 초기 /task/status를 아직 받지 못했으면 이동/모드 버튼을 모두 비활성화한다.
        - safety_state가 NORMAL이 아니면 AUTO/MANUAL 전환과 pose 이동 버튼을
          비활성화한다(복구 요청 버튼은 그대로 유지한다).
        - 작업이 진행 중(state != IDLE)이면 중복 명령을 막기 위해 이동/모드 버튼을
          비활성화한다.
        - MANUAL 모드가 아니면 pose 이동 버튼을 비활성화한다(AUTO/MANUAL 전환
          버튼 자체는 현재 모드와 무관하게 위 조건만 만족하면 활성 상태를 유지한다 -
          이미 그 모드로 전환해도 task_manager가 안전하게 무시한다).
        - 작업 중단(STOP)과 복구 요청(RESET) 버튼은 이 정책과 무관하게 항상
          활성화된 상태로 둔다.
        """
        if not self._received_first_status:
            for button in (self.auto_button, self.manual_button, self.home_button,
                           self.front_button, self.up_button, self.down_button,
                           self.watch_button):
                button.setEnabled(False)
            return

        normal = self._last_safety_state == 'NORMAL'
        busy = self._last_state not in (None, 'IDLE')
        is_manual = self._last_operation_mode == 'MANUAL'

        mode_switch_enabled = normal and not busy
        self.auto_button.setEnabled(mode_switch_enabled)
        self.manual_button.setEnabled(mode_switch_enabled)

        pose_buttons_enabled = normal and not busy and is_manual
        for button in (self.home_button, self.front_button, self.up_button,
                       self.down_button, self.watch_button):
            button.setEnabled(pose_buttons_enabled)

        # stop_button/reset_button은 항상 활성화 상태를 유지한다(정책 대상에서 제외).

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
        glow, tint = SAFETY_STATE_COLORS.get(severity, DEFAULT_SAFETY_COLOR)
        self.safety_label.setText(f'안전상태: {severity}')
        self._style_safety_chip(self.safety_label, glow, tint)
        self._apply_safety_pulse(severity)

        self.fault_banner.setText(message)
        self.fault_banner.setStyleSheet(
            f'background-color: {tint}; color: {glow}; '
            f'font-family: {_MONO_FONT_STACK}; font-weight: 700; '
            f'border: 1px solid {glow}; border-radius: 3px; padding: 6px 10px;')
        self.fault_banner.show()
        self._log(f'Fault: {message}')
        # 다음 /task/status를 기다리지 않고 즉시 이동/AUTO/MANUAL 버튼을 차단한다
        # (safety_state는 위에서 이미 비정상으로 갱신했다). STOP/RESET은 정책 대상이
        # 아니므로 그대로 활성 상태를 유지한다.
        self._apply_button_policy()

    def _update_connection_status(self, connected):
        if connected:
            self.connection_label.setText('ROS2: 연결됨')
            self.connection_label.setStyleSheet(
                f'font-family: {_MONO_FONT_STACK}; font-size: 12px; font-weight: 700; '
                'padding: 6px 12px; border: 1px solid #39e991; border-radius: 3px; '
                'background-color: rgba(57, 233, 145, 0.14); color: #39e991;')
        else:
            self.connection_label.setText('ROS2: 연결 안 됨')
            self.connection_label.setStyleSheet(
                f'font-family: {_MONO_FONT_STACK}; font-size: 12px; font-weight: 700; '
                'padding: 6px 12px; border: 1px solid #ff4d5e; border-radius: 3px; '
                'background-color: rgba(255, 77, 94, 0.18); color: #ff4d5e;')
        self._log('ROS2 연결됨' if connected else 'ROS2 연결 끊김')

    # ---- 카메라 ----

    def _update_camera_image(self, image_bytes):
        image = QtGui.QImage.fromData(image_bytes)
        if image.isNull():
            return
        self._camera_image = image
        self._last_camera_image_at = time.monotonic()
        if self._camera_stale:
            # 새 영상이 다시 도착했으니 "멈춤" 표시를 자동으로 해제한다.
            self._camera_stale = False
            self._log('카메라 영상이 다시 수신되기 시작했습니다.')
        self._render_camera_pixmap()

    def _check_camera_stale(self):
        """QTimer가 주기적으로 호출한다(GUI 메인 스레드) - ROS 콜백 스레드에서 직접
        위젯을 건드리지 않고, 여기서만 카메라 라벨을 갱신한다."""
        if self._last_camera_image_at is None:
            return  # 아직 한 번도 영상을 받지 못함 - "카메라 대기 중..." 문구를 유지한다.
        if self._camera_stale:
            return
        elapsed = time.monotonic() - self._last_camera_image_at
        if elapsed > self.camera_stale_timeout_s:
            self._camera_stale = True
            self.camera_label.setPixmap(QtGui.QPixmap())
            self.camera_label.setText('카메라 영상이 멈췄습니다.')
            self._log('카메라 영상이 멈췄습니다.')

    def _render_camera_pixmap(self):
        if self._camera_stale:
            return  # 멈춘 상태에서는 마지막 프레임을 다시 그리지 않는다(텍스트 유지).
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
        self._camera_stale_timer.stop()
        self._safety_pulse.stop()
        try:
            self.ros_client.close()
        except Exception:
            pass
        super().closeEvent(event)
