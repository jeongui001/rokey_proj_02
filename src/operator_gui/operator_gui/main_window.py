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

# 파이프라인 점검.md의 Phase A~K 체크리스트와 1:1 대응하는 체크포인트 정의.
# (phase, checkpoint_id, 화면에 보여줄 설명, 자동 판정 여부)
CHECKPOINTS = [
    ('A', 'stt_utterance_published', '발화 텍스트가 그대로 퍼블리시되는가', True),
    ('A', 'stt_recording_not_truncated', '5초 고정 녹음이라 말이 잘리지 않는가', False),
    ('B', 'parse_no_intermediate_state', 'command_parser가 즉시 판정하는가(중간 PARSING 없음)', True),
    ('B', 'move_watch_goal_sent', 'move_named(watch) 골이 전송되는가', True),
    ('B', 'move_watch_robot_moved', '로봇이 실제로 watch 자세로 이동하는가', False),
    ('B', 'move_watch_result_received', '이동 완료 결과가 task_manager로 돌아오는가', True),
    ('C', 'vision_set_mode_track_tool', '/vision/set_mode(TRACK_TOOL) 응답이 success인가', True),
    ('C', 'tool_track_valid', 'ToolTrack(위치/뎁스 유효/접근 여부)이 그럴듯한가', True),
    ('C', 'servo_pick_triggered', '트리거 게이트 통과 후 SERVO_PICK으로 전이되는가', True),
    ('D', 'servo_pick_state_entered', '/task/status에서 SERVO_PICK 진입 확인', True),
    ('D', 'servo_tracking_followed', '서보 루프가 공구를 따라가는가', False),
    ('D', 'gripper_closed', '그리퍼가 실제로 닫히는가', True),
    ('D', 'servo_pick_result', 'success/final_width_mm/grip_detected가 정상 반환되는가', True),
    ('E', 'grasp_verified', 'grip_detected + 폭 범위로 재확인하는가', True),
    ('E', 'move_safe_entered', 'MOVE_SAFE로 전이되는가', True),
    ('F', 'handover_safe_goal_sent', 'move_named(handover_safe) 골 전송 확인', True),
    ('F', 'handover_safe_robot_moved', '로봇이 handover_safe 자세로 이동하는가', False),
    ('F', 'handover_safe_result_received', '이동 완료 결과 수신 확인', True),
    ('G', 'approach_hand_entered', '/task/status에서 APPROACH_HAND 전이 확인', True),
    ('G', 'vision_set_mode_track_hand', '/vision/set_mode(TRACK_HAND) 호출 및 success 응답', True),
    ('H', 'hand_pose_published', '/vision/hand_pose가 퍼블리시되는가', True),
    ('H', 'handover_approach_goal_sent', 'handover_approach 골 전송 확인', True),
    ('H', 'handover_approach_robot_moved', '로봇이 손 쪽으로 이동하는가', False),
    ('H', 'handover_approach_result_received', '이동 완료 결과 수신 확인', True),
    ('I', 'wait_pull_entered', 'WAIT_PULL 전이 확인', True),
    ('I', 'handover_hold_goal_sent', 'handover_hold 골 전송 확인', True),
    ('I', 'compliance_mode_active', '컴플라이언스 모드가 가동되는가', True),
    ('I', 'gripper_opened_on_pull', '공구를 당겼을 때 그리퍼가 개방되는가', True),
    ('I', 'compliance_mode_ended', '개방 후 컴플라이언스가 종료되는가', True),
    ('I', 'handover_hold_result_received', '결과 수신 확인', True),
    ('J', 'home_goal_sent', 'move_named(home) 골 전송 확인', True),
    ('J', 'gripper_auto_opened_before_home', '그리퍼가 열려있지 않으면 자동으로 open하는가', False),
    ('J', 'home_result_received', 'home 이동 완료 결과 수신 확인', True),
    ('K', 'vision_set_mode_off', '/vision/set_mode(OFF) 호출 및 success 응답', True),
    ('K', 'idle_entered', '/task/status가 IDLE로 돌아오는가', True),
    ('K', 'second_cycle_started', '이후 다시 발화했을 때 2회차 사이클이 정상 시작되는가', False),
]
PHASE_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
PHASE_TITLES = {
    'A': 'Phase A — STT',
    'B': 'Phase B — 파싱 → MOVE_TO_WATCH',
    'C': 'Phase C — 공구 추적',
    'D': 'Phase D — servo_pick 파지',
    'E': 'Phase E — 파지 검증 → MOVE_SAFE',
    'F': 'Phase F — handover_safe 이동',
    'G': 'Phase G — TRACK_HAND 전환',
    'H': 'Phase H — 손 위치 추적~접근',
    'I': 'Phase I — handover_hold 당김 감지',
    'J': 'Phase J — HOME 복귀',
    'K': 'Phase K — vision OFF ~ IDLE',
}

_STATUS_ICON = {'PENDING': '⏳', 'PASS': '✅', 'FAIL': '❌'}
_STATUS_COLOR = {'PENDING': '#7f95a3', 'PASS': '#39e991', 'FAIL': '#ff4d5e'}


class _CheckpointRow(QtWidgets.QWidget):
    """파이프라인 점검 체크리스트의 한 줄. 자동 판정 항목은 ROS 이벤트로만
    상태가 바뀌고, 수동 항목은 체크박스를 눌러 오퍼레이터가 직접 표시한다."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, checkpoint_id, label_text, auto, parent=None):
        super().__init__(parent)
        self.checkpoint_id = checkpoint_id
        self.label_text = label_text
        self.auto = auto
        self.status = 'PENDING'
        self.payload = None

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self.status_label = QtWidgets.QLabel(_STATUS_ICON['PENDING'])
        self.status_label.setFixedWidth(20)
        self.text_label = QtWidgets.QLabel(label_text)
        self.text_label.setWordWrap(True)
        # 드래그로 문구를 선택해 Ctrl+C로 복사할 수 있게 한다(QLabel 기본 기능).
        self.text_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)
        layout.addWidget(self.text_label, stretch=1)
        self.checkbox = None
        if not auto:
            self.checkbox = QtWidgets.QCheckBox()
            self.checkbox.toggled.connect(self._on_checkbox_toggled)
            layout.addWidget(self.checkbox)
        self._apply_style()

    def _on_checkbox_toggled(self, checked):
        self.set_status('PASS' if checked else 'PENDING')

    def set_status(self, status, payload=None):
        self.status = status
        self.payload = payload
        self._apply_style()

    def _apply_style(self):
        self.status_label.setText(_STATUS_ICON[self.status])
        self.text_label.setStyleSheet(f'color: {_STATUS_COLOR[self.status]};')

    def reset(self):
        self.payload = None
        if self.checkbox is not None:
            self.checkbox.blockSignals(True)
            self.checkbox.setChecked(False)
            self.checkbox.blockSignals(False)
        self.set_status('PENDING')

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


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
        self._right_compact_scale = None

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
        self.right_container = right_container
        self._right_group_layouts = [
            mode_layout, pose_grid, estop_layout, free_command_layout, mic_layout, right_panel]
        self._right_buttons = [
            self.auto_button, self.manual_button,
            self.home_button, self.front_button, self.watch_button, self.up_button, self.down_button,
            self.stop_button, self.reset_button, self.resume_button,
            self.send_button,
        ]
        self._right_groups = [mode_group, move_group, estop_group, cmd_group, mic_group]

        # 가장 중요한 정보이므로 카메라와 비슷한 비중으로 항상 크게 노출한다(토글 없음).
        self.debug_panel = QtWidgets.QGroupBox('오류 확인')
        self._checkpoint_rows = {}
        scroll_content = QtWidgets.QWidget()
        scroll_content_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_content_layout.setSpacing(6)
        for phase in PHASE_ORDER:
            phase_group = QtWidgets.QGroupBox(PHASE_TITLES[phase])
            phase_layout = QtWidgets.QVBoxLayout()
            for cp_phase, checkpoint_id, label, auto in CHECKPOINTS:
                if cp_phase != phase:
                    continue
                row = _CheckpointRow(checkpoint_id, label, auto)
                row.clicked.connect(
                    lambda checkpoint_id=checkpoint_id: self._show_checkpoint_detail(checkpoint_id))
                self._checkpoint_rows[checkpoint_id] = row
                phase_layout.addWidget(row)
            phase_group.setLayout(phase_layout)
            scroll_content_layout.addWidget(phase_group)
        scroll_content_layout.addStretch(1)
        self.checklist_scroll = QtWidgets.QScrollArea()
        self.checklist_scroll.setWidgetResizable(True)
        self.checklist_scroll.setWidget(scroll_content)
        self.checklist_scroll.setMinimumHeight(220)

        self.reset_checklist_button = QtWidgets.QPushButton('초기화')
        self.reset_checklist_button.clicked.connect(self._reset_checklist)
        debug_header = QtWidgets.QHBoxLayout()
        debug_header.addStretch(1)
        debug_header.addWidget(self.reset_checklist_button)
        debug_layout = QtWidgets.QVBoxLayout()
        debug_layout.addLayout(debug_header)
        debug_layout.addWidget(self.checklist_scroll)
        self.debug_panel.setLayout(debug_layout)

        # 하단: 시간순 로그
        self.log_view = QtWidgets.QListWidget()
        self.log_view.setMinimumHeight(90)

        left_column = QtWidgets.QVBoxLayout()
        left_column.addWidget(self.camera_label, stretch=1)
        left_column.addWidget(self.debug_panel, stretch=1)
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
        self._apply_right_compact_scale()

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

    # ---- 체크리스트 ----

    def _reset_checklist(self):
        for row in self._checkpoint_rows.values():
            row.reset()

    def _update_debug_event(self, payload):
        if not isinstance(payload, dict):
            return
        status = payload.get('status')
        if status not in ('PASS', 'FAIL'):
            return
        row = self._checkpoint_rows.get(payload.get('checkpoint_id'))
        if row is None:
            return
        row.set_status(status, payload)
        self._log(f"체크포인트 갱신: {row.checkpoint_id} -> {status}")

    def _show_checkpoint_detail(self, checkpoint_id):
        row = self._checkpoint_rows.get(checkpoint_id)
        if row is None:
            return
        if row.payload is None:
            QtWidgets.QMessageBox.information(
                self, '오류 확인', f'{row.label_text}\n\n(아직 수신된 이벤트가 없습니다.)')
            return
        payload = row.payload
        data = payload.get('data') or {}
        detail = f"[{payload.get('status')}] {row.label_text} - {payload.get('message', '')}"
        if data:
            detail += '\n\n수치:\n' + '\n'.join(
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
        self._apply_right_compact_scale()
        self._render_camera_pixmap()
        self._reposition_camera_overlay()

    def _apply_right_compact_scale(self):
        """창 높이가 낮을 때 우측 메뉴 전체를 같은 비율로 줄인다.

        Qt 기본 레이아웃 압축에 맡기면 일부 버튼만 먼저 납작해져 보이므로,
        버튼 높이/폰트/그룹 여백을 한 스케일로 맞춘다.
        """
        if not hasattr(self, 'right_container'):
            return
        scale = max(0.72, min(1.0, self.height() / 720.0))
        bucket = round(scale, 2)
        if self._right_compact_scale == bucket:
            return
        self._right_compact_scale = bucket

        font_px = max(10, int(13 * scale))
        title_px = max(9, int(11 * scale))
        button_h = max(24, int(40 * scale))
        input_h = max(28, int(38 * scale))
        padding_v = max(3, int(8 * scale))
        padding_h = max(7, int(12 * scale))
        group_top = max(10, int(16 * scale))
        group_pad_v = max(6, int(10 * scale))
        group_pad_h = max(7, int(10 * scale))
        spacing = max(4, int(8 * scale))

        for button in self._right_buttons:
            button.setMinimumHeight(button_h)
            button.setMaximumHeight(button_h)
        self.command_input.setMinimumHeight(input_h)
        self.command_input.setMaximumHeight(input_h)
        self.mic_level_bar.setFixedHeight(max(10, int(16 * scale)))
        self.stt_command_label.setMaximumHeight(max(26, int(34 * scale)))

        for layout in self._right_group_layouts:
            layout.setSpacing(spacing)
            if isinstance(layout, QtWidgets.QGridLayout):
                layout.setContentsMargins(spacing, spacing, spacing, spacing)
            else:
                layout.setContentsMargins(spacing, spacing, spacing, spacing)
        self.right_container.layout().setContentsMargins(0, 0, 0, 0)

        self.right_container.setStyleSheet(f"""
QGroupBox {{
    margin-top: {group_top}px;
    padding: {group_pad_v}px {group_pad_h}px {group_pad_v}px {group_pad_h}px;
    font-size: {title_px}px;
}}
QGroupBox::title {{
    top: 1px;
    left: {group_pad_h}px;
    padding: 0 5px;
    font-size: {title_px}px;
}}
QPushButton {{
    padding: {padding_v}px {padding_h}px;
    font-size: {font_px}px;
}}
QLineEdit {{
    padding: {padding_v}px {padding_h}px;
    font-size: {font_px}px;
}}
QLabel {{
    font-size: {max(10, int(12 * scale))}px;
}}
""")

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
