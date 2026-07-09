import math
import os
import time
from collections import deque

from PyQt5 import QtCore, QtGui, QtWidgets

# 카메라 프레임이 이 시간(초) 이상 없으면 "멈췄습니다"로 표시한다. 실제 카메라
# 프레임 주기에 맞춰 조정 가능한 통신 타이밍 값이며, 하드웨어 확정값이 아니다.
DEFAULT_CAMERA_STALE_TIMEOUT_S = 2.0
# 위 timeout과 무관하게, 화면 갱신 자체를 확인하는 주기(ms). timeout보다 충분히
# 짧게 유지해 멈춤 표시가 늦게 뜨지 않게 한다.
_CAMERA_STALE_CHECK_INTERVAL_MS = 300
_MAX_DEBUG_EVENTS = 200

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
QPushButton#warningButton {{
    background-color: transparent;
    color: #ffd166;
    border: 1px solid #ffd166;
    font-weight: 700;
}}
QPushButton#warningButton:checked {{
    background-color: rgba(255, 209, 102, 0.18);
    color: #ffd166;
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
CMD_RESUME = '재개'

# 카메라 영상 위 오버레이용 짧은 한글 표시 - task_manager.task_models.State/Safety와
# 1:1 대응. 여기 없는 값(향후 새 state 추가 등)은 원래 문자열을 그대로 보여준다
# (오버레이가 새 state를 몰라서 죽거나 빈 칸이 되지 않도록).
STATE_OVERLAY_LABELS = {
    'IDLE': '대기중',
    'MOVE_TO_WATCH': '이동 중',
    'DETECT_TRACK': '탐지 중',
    'SERVO_PICK': '집는 중',
    'VERIFY_GRASP': '파지 확인 중',
    'MOVE_SAFE': '이동 중',
    'APPROACH_HAND': '전달 접근 중',
    'WAIT_PULL': '전달 대기 중',
    'HOME': '홈 이동 중',
    'MANUAL_MOVE': '수동 이동 중',
    'CANCELLING': '취소 중',
}
# safety_state가 NORMAL이 아니면 작업 state보다 이걸 우선 표시한다(더 시급한 정보).
SAFETY_OVERLAY_LABELS = {
    'PROTECTIVE_STOP': '보호정지',
    'RECOVERY_REQUIRED': '복구 필요',
    'EMERGENCY_STOP': '비상정지',
    'FAULT': '오류',
}


class MainWindow(QtWidgets.QMainWindow):
    # state, detail, operation_mode, safety_state, resumable
    task_status_received = QtCore.pyqtSignal(str, str, str, str, bool)
    gripper_state_received = QtCore.pyqtSignal(float, bool)
    fault_received = QtCore.pyqtSignal(str)
    connection_changed = QtCore.pyqtSignal(bool)
    camera_image_received = QtCore.pyqtSignal(bytes)
    mic_level_received = QtCore.pyqtSignal(float)
    stt_status_received = QtCore.pyqtSignal(str, str, object)
    stt_command_received = QtCore.pyqtSignal(str)
    debug_event_received = QtCore.pyqtSignal(object)

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
        self._last_resumable = False
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
        self._debug_events = deque(maxlen=_MAX_DEBUG_EVENTS)
        self._unseen_debug_count = 0

        self._build_ui()
        self._wire_signals()
        self._apply_button_policy()
        self._reposition_camera_overlay()

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
        self.detail_label.setMinimumWidth(0)
        self.detail_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.safety_label.setStyleSheet(_chip_base)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(8)
        top_bar.addWidget(self.connection_label)
        top_bar.addWidget(self.mode_label)
        top_bar.addWidget(self.state_label)
        top_bar.addWidget(self.safety_label)
        top_bar.addWidget(self.detail_label, stretch=1)
        self.debug_toggle_button = QtWidgets.QPushButton('오류 확인')
        self.debug_toggle_button.setObjectName('warningButton')
        self.debug_toggle_button.setCheckable(True)
        self.debug_toggle_button.setMinimumWidth(76)
        self.debug_toggle_button.clicked.connect(self._toggle_debug_panel)
        top_bar.addWidget(self.debug_toggle_button)

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
        self.camera_label.setMinimumSize(280, 200)

        # 카메라 영상 안쪽 우측 상단에 떠 있는 상태 오버레이. camera_label의 레이아웃
        # 자식이 아니라 좌표를 직접 move()하는 "떠 있는" 자식 위젯이라, 카메라가
        # 대기/멈춤 텍스트를 보여줄 때도(즉 pixmap이 없을 때도) 항상 그 위에 그려진다.
        self.camera_overlay_label = QtWidgets.QLabel('-', self.camera_label)
        self.camera_overlay_label.setStyleSheet(
            f'font-family: {_MONO_FONT_STACK}; font-size: 12px; font-weight: 700; '
            'padding: 4px 10px; border-radius: 3px; '
            'background-color: rgba(5, 8, 13, 0.72); color: #cfe6f2;')
        self.camera_overlay_label.adjustSize()

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
        self.resume_button = QtWidgets.QPushButton('재개')
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(lambda: self._send_text(CMD_RESUME))
        self.estop_notice_label = QtWidgets.QLabel(
            '※ 실제 비상정지(E-Stop)는 로봇 본체의 물리 버튼입니다.')
        self.estop_notice_label.setObjectName('sectionNote')
        self.estop_notice_label.setWordWrap(True)
        estop_layout = QtWidgets.QVBoxLayout()
        estop_layout.addWidget(self.stop_button)
        estop_layout.addWidget(self.reset_button)
        estop_layout.addWidget(self.resume_button)
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

        # 마이크가 소리를 잘 잡고 있는지 눈으로 확인하기 위한 용도라 절대 dB 눈금은
        # 표시하지 않는다 - 막대가 조용할 때 낮고 말할 때 올라가는지만 보면 된다.
        # 게이지+라벨을 세로로 쌓지 않고 가로 한 줄로 배치해 그룹박스 세로 폭을
        # 아낀다 - 우측 패널에 그룹박스가 이미 여러 개 쌓여있어 창 최소 높이가
        # 화면 높이에 근접하면 최대화 버튼이 사실상 동작하지 않게 된다.
        mic_group = QtWidgets.QGroupBox('음성 인식')
        self.mic_level_bar = QtWidgets.QProgressBar()
        self.mic_level_bar.setRange(0, 100)
        self.mic_level_bar.setTextVisible(False)
        self.mic_level_bar.setFixedHeight(16)
        self.stt_command_label = QtWidgets.QLabel('마지막 명령어: -')
        self.stt_command_label.setWordWrap(True)
        self.stt_command_label.setMaximumHeight(34)  # 최대 2줄 - 긴 명령어도 그룹박스 높이를 무한정 늘리지 않는다
        self.stt_command_label.setStyleSheet(f'font-family: {_MONO_FONT_STACK}; font-size: 12px;')
        self.stt_status_label = QtWidgets.QLabel('대기 중')
        self.stt_status_label.setStyleSheet('color: #ffd166; font-size: 12px; font-weight: 600;')
        mic_layout = QtWidgets.QVBoxLayout()
        mic_layout.addWidget(self.stt_status_label)
        mic_layout.addWidget(self.mic_level_bar)
        mic_layout.addWidget(self.stt_command_label, stretch=1)
        mic_group.setLayout(mic_layout)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(10)
        right_panel.addWidget(self.gripper_label)
        right_panel.addWidget(mode_group)
        right_panel.addWidget(move_group)
        right_panel.addWidget(estop_group)
        right_panel.addStretch(1)
        right_panel.addWidget(cmd_group)
        right_panel.addWidget(mic_group)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        right_container.setMinimumWidth(320)
        right_container.setMaximumWidth(360)

        # 하단: 시간순 로그 - 최소 10줄 이상 보이도록 높이를 확보한다.
        self.debug_panel = QtWidgets.QGroupBox('오류 확인')
        self.debug_log_view = QtWidgets.QListWidget()
        self.debug_log_view.setMinimumHeight(80)
        self.debug_log_view.itemClicked.connect(self._show_debug_event_detail)
        self.clear_debug_button = QtWidgets.QPushButton('비우기')
        self.clear_debug_button.clicked.connect(self._clear_debug_events)
        debug_header = QtWidgets.QHBoxLayout()
        debug_header.addStretch(1)
        debug_header.addWidget(self.clear_debug_button)
        debug_layout = QtWidgets.QVBoxLayout()
        debug_layout.addLayout(debug_header)
        debug_layout.addWidget(self.debug_log_view)
        self.debug_panel.setLayout(debug_layout)
        self.debug_panel.setMaximumHeight(190)
        self.debug_panel.hide()

        self.log_view = QtWidgets.QListWidget()
        self.log_view.setMinimumHeight(90)

        left_column = QtWidgets.QVBoxLayout()
        left_column.addWidget(self.camera_label, stretch=1)
        left_column.addWidget(self.debug_panel)
        left_column.addWidget(self.log_view, stretch=0)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_column)

        middle_layout = QtWidgets.QHBoxLayout()
        middle_layout.addWidget(left_widget, stretch=1)
        middle_layout.addWidget(right_container, stretch=0)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.fault_banner)
        layout.addLayout(middle_layout, stretch=1)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(960, 680)
        self.setMinimumSize(720, 480)

    def _wire_signals(self):
        self.task_status_received.connect(self._update_task_status)
        self.gripper_state_received.connect(self._update_gripper_state)
        self.fault_received.connect(self._update_fault)
        self.connection_changed.connect(self._update_connection_status)
        self.camera_image_received.connect(self._update_camera_image)
        self.mic_level_received.connect(self._update_mic_level)
        self.stt_status_received.connect(self._update_stt_status)
        self.stt_command_received.connect(self._update_stt_command)
        self.debug_event_received.connect(self._update_debug_event)

        self.ros_client.on_task_status = self.task_status_received.emit
        self.ros_client.on_gripper_state = self.gripper_state_received.emit
        self.ros_client.on_fault = self.fault_received.emit
        self.ros_client.on_connection_changed = self.connection_changed.emit
        self.ros_client.on_camera_image = self.camera_image_received.emit
        self.ros_client.on_mic_level = self.mic_level_received.emit
        self.ros_client.on_stt_status = self.stt_status_received.emit
        self.ros_client.on_stt_command = self.stt_command_received.emit
        self.ros_client.on_debug_event = self.debug_event_received.emit

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

    def _update_task_status(self, state, detail, operation_mode, safety_state, resumable=False):
        self.state_label.setText(f'상태: {state}')
        self.detail_label.setText(f'디테일: {detail}')
        self.mode_label.setText(f'모드: {operation_mode or "-"}')

        if self._last_state is not None and state != self._last_state:
            self._log(f'작업 상태 변화: {self._last_state} -> {state}')
        self._last_state = state

        if self._last_operation_mode is not None and operation_mode != self._last_operation_mode:
            self._log(f'모드 전환: {self._last_operation_mode} -> {operation_mode}')
        self._last_operation_mode = operation_mode

        self._last_resumable = bool(resumable)

        self._apply_safety_state(safety_state)
        self._received_first_status = True
        self._apply_button_policy()
        self._update_camera_overlay()

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
        - 재개(RESUME) 버튼은 task_manager가 /task/status.resumable로 보고해준
          값(안전상태 NORMAL + state IDLE + 재개할 스냅샷 존재)을 그대로 따른다 -
          최종 판단은 task_manager가 하므로 여기서는 그 값을 그대로 반영할 뿐이다.
        """
        if not self._received_first_status:
            for button in (self.auto_button, self.manual_button, self.home_button,
                           self.front_button, self.up_button, self.down_button,
                           self.watch_button, self.resume_button):
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

        self.resume_button.setEnabled(bool(self._last_resumable))

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
        self._update_camera_overlay()

    # ---- 카메라 오버레이 (우측 상단, 로봇 현재 상태) ----

    def _update_camera_overlay(self):
        """카메라 영상이 대기/멈춤 상태라도(즉 pixmap이 없어도) 로봇이 지금 뭘
        하고 있는지는 계속 보여준다 - 안전상태가 NORMAL이 아니면 작업 state보다
        그걸 우선 표시한다(더 시급한 정보이므로)."""
        if not self._received_first_status:
            text = '-'
        elif self._last_safety_state and self._last_safety_state != 'NORMAL':
            text = SAFETY_OVERLAY_LABELS.get(self._last_safety_state, self._last_safety_state)
        else:
            text = STATE_OVERLAY_LABELS.get(self._last_state, self._last_state or '-')
        self.camera_overlay_label.setText(text)
        self._reposition_camera_overlay()

    def _reposition_camera_overlay(self):
        margin = 8
        self.camera_overlay_label.adjustSize()
        x = self.camera_label.width() - self.camera_overlay_label.width() - margin
        self.camera_overlay_label.move(max(x, margin), margin)

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

    # ---- 음성 인식 ----

    def _update_mic_level(self, level):
        self.mic_level_bar.setValue(int(min(max(level, 0.0), 1.0) * 100))

    def _update_stt_status(self, state, detail, data):
        text = detail or state or '대기 중'
        self.stt_status_label.setText(text)
        if state == 'wakeword_detected':
            self._log(text)
        elif state in ('error', 'silent_skipped'):
            self._log(f'음성 인식: {text}')

    def _update_stt_command(self, text):
        self.stt_command_label.setText(f'마지막 명령어: {text}')

    # ---- 디버그 이벤트 ----

    def _toggle_debug_panel(self, checked):
        self.debug_panel.setVisible(bool(checked))
        if checked:
            self._unseen_debug_count = 0
            self._refresh_debug_button_text()

    def _clear_debug_events(self):
        self._debug_events.clear()
        self.debug_log_view.clear()
        self._unseen_debug_count = 0
        self._refresh_debug_button_text()

    def _refresh_debug_button_text(self):
        if self._unseen_debug_count > 0:
            self.debug_toggle_button.setText(f'오류 확인 ({self._unseen_debug_count})')
        else:
            self.debug_toggle_button.setText('오류 확인')

    @staticmethod
    def _format_debug_event(payload):
        node = payload.get('node', '-')
        level = payload.get('level', '-')
        category = payload.get('category', '-')
        reason = payload.get('reason', '-')
        message = payload.get('message', '')
        data = payload.get('data') or {}
        compact_data = ''
        if data:
            pairs = []
            for key in sorted(data.keys())[:4]:
                value = data[key]
                if isinstance(value, float):
                    value = f'{value:.4g}'
                pairs.append(f'{key}={value}')
            compact_data = ' | ' + ', '.join(pairs)
        return f'[{level}] {node}/{category} reason={reason} {message}{compact_data}'

    def _update_debug_event(self, payload):
        if not isinstance(payload, dict):
            return
        level = payload.get('level', '')
        if level not in ('WARN', 'ERROR', 'FAULT'):
            return
        self._debug_events.append(payload)
        item = QtWidgets.QListWidgetItem(self._format_debug_event(payload))
        if level == 'WARN':
            item.setForeground(QtGui.QColor('#ffd166'))
        else:
            item.setForeground(QtGui.QColor('#ff4d5e'))
        item.setData(QtCore.Qt.UserRole, payload)
        self.debug_log_view.addItem(item)
        while self.debug_log_view.count() > _MAX_DEBUG_EVENTS:
            self.debug_log_view.takeItem(0)
        self.debug_log_view.scrollToBottom()
        if not self.debug_panel.isVisible():
            self._unseen_debug_count += 1
            self._refresh_debug_button_text()
        self._log(f'디버그 이벤트: {self._format_debug_event(payload)}')

    def _show_debug_event_detail(self, item):
        payload = item.data(QtCore.Qt.UserRole)
        if not isinstance(payload, dict):
            return
        data = payload.get('data') or {}
        detail = self._format_debug_event(payload)
        if data:
            detail = f'{detail}\n\n수치:\n' + '\n'.join(
                f'- {key}: {value}' for key, value in sorted(data.items()))
        QtWidgets.QMessageBox.information(self, '오류 확인', detail)

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
        self._reposition_camera_overlay()

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
