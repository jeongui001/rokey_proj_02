import threading

import pytest
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt

from operator_gui.main_window import CMD_RESUME, DEFAULT_CAMERA_STALE_TIMEOUT_S, MainWindow


class _FakeRosClient:
    def __init__(self, connected=True):
        self.published = []
        self.on_task_status = None
        self.on_gripper_state = None
        self.on_fault = None
        self.on_connection_changed = None
        self.on_camera_image = None
        self.reconnect_interval_s = 5.0
        self._connected = connected
        self.ensure_connected_calls = 0
        self.closed = False

    def is_connected(self):
        return self._connected

    def publish_command(self, text):
        text = (text or '').strip()
        if not text or not self._connected:
            return False
        self.published.append(text)
        return True

    def ensure_connected(self):
        self.ensure_connected_calls += 1

    def close(self):
        self.closed = True


@pytest.fixture
def window(qtbot):
    ros_client = _FakeRosClient()
    win = MainWindow(ros_client)
    qtbot.addWidget(win)
    win.show()
    return win


def _png_bytes():
    image = QtGui.QImage(4, 4, QtGui.QImage.Format_RGB32)
    image.fill(Qt.red)
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QBuffer.ReadWrite)
    image.save(buffer, 'PNG')
    return bytes(buffer.data())


# ---- /task/status 4개 필드 파싱 ----

def test_task_status_updates_all_top_labels(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('MOVE_TO_WATCH', '감시 자세로 이동', 'AUTO', 'NORMAL', False)

    assert window.state_label.text() == '상태: MOVE_TO_WATCH'
    assert window.detail_label.text() == '디테일: 감시 자세로 이동'
    assert window.mode_label.text() == '모드: AUTO'
    assert window.safety_label.text() == '안전상태: NORMAL'


def test_task_state_change_is_logged_but_first_report_is_not(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL', False)
    count_after_first = window.log_view.count()

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('MOVE_TO_WATCH', '', 'AUTO', 'NORMAL', False)

    assert window.log_view.count() > count_after_first


# ---- AUTO/MANUAL, 명령 버튼 (parametrize) ----

@pytest.mark.parametrize('button_attr,expected_text', [
    ('auto_button', '자동 모드로 전환해'),
    ('manual_button', '수동 모드로 전환해'),
    ('home_button', '홈으로 가'),
    ('front_button', '정면을 봐'),
    ('up_button', '위를 봐'),
    ('down_button', '아래를 봐'),
    ('watch_button', '컨베이어를 봐'),
    ('stop_button', '멈춰'),
    ('reset_button', '리셋'),
])
def test_fixed_command_buttons_publish_expected_text(window, qtbot, button_attr, expected_text):
    # 초기 /task/status를 받기 전에는 이동/모드 버튼이 비활성화되므로, 먼저 정상
    # 상태(MANUAL/NORMAL/IDLE)를 수신한 것으로 시뮬레이션한다.
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL', False)

    button = getattr(window, button_attr)
    qtbot.mouseClick(button, Qt.LeftButton)

    assert window.ros_client.published == [expected_text]


def test_stop_button_label_does_not_mention_emergency_or_estop(window):
    label = window.stop_button.text()
    assert '비상' not in label
    assert 'E-Stop' not in label
    assert 'e-stop' not in label.lower()
    assert '실제 비상정지' in window.estop_notice_label.text()


def test_send_button_publishes_command_and_clears_input(window, qtbot):
    window.command_input.setText('스패너 갖다줘')
    qtbot.mouseClick(window.send_button, Qt.LeftButton)

    assert window.ros_client.published == ['스패너 갖다줘']
    assert window.command_input.text() == ''


def test_send_button_ignores_empty_input(window, qtbot):
    window.command_input.setText('   ')
    qtbot.mouseClick(window.send_button, Qt.LeftButton)

    assert window.ros_client.published == []


# ---- 연결 안 됐을 때 명령 차단 ----

def test_command_blocked_when_not_connected(qtbot):
    ros_client = _FakeRosClient(connected=False)
    win = MainWindow(ros_client)
    qtbot.addWidget(win)
    win.show()
    with qtbot.waitSignal(win.task_status_received, timeout=1000):
        win.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL', False)

    qtbot.mouseClick(win.home_button, Qt.LeftButton)

    assert ros_client.published == []
    assert win.log_view.count() >= 1
    assert '실패' in win.log_view.item(win.log_view.count() - 1).text()


# ---- Fault 배너 상태 ----

@pytest.mark.parametrize('safety_state,expected_color', [
    ('NORMAL', '#39e991'),
    ('PROTECTIVE_STOP', '#ffb454'),
    ('RECOVERY_REQUIRED', '#ffb454'),
    ('EMERGENCY_STOP', '#ff4d5e'),
    ('FAULT', '#ff4d5e'),
])
def test_safety_state_label_color(window, qtbot, safety_state, expected_color):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', safety_state, False)

    assert expected_color in window.safety_label.styleSheet()


def test_fault_banner_shows_message(window, qtbot):
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('torque anomaly')

    assert window.fault_banner.isVisible()
    assert 'torque anomaly' in window.fault_banner.text()


def test_fault_immediately_disables_move_and_mode_buttons_without_waiting_for_status(window, qtbot):
    # 1) MANUAL/NORMAL/IDLE로 pose 버튼이 활성화된 상태를 준비한다.
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL', False)
    assert window.home_button.isEnabled() is True
    assert window.auto_button.isEnabled() is True

    # 2) /task/status 추가 메시지 없이 /robot/fault만 직접 수신한다.
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('torque anomaly')

    # 3) 다음 /task/status를 기다리지 않고 즉시 이동/모드 버튼이 비활성화되어야 한다.
    for attr in ('auto_button', 'manual_button', 'home_button', 'front_button',
                 'up_button', 'down_button', 'watch_button'):
        assert getattr(window, attr).isEnabled() is False, attr
    # 4) STOP/RESET은 그대로 활성 상태를 유지한다.
    assert window.stop_button.isEnabled() is True
    assert window.reset_button.isEnabled() is True


def test_fault_banner_persists_until_normal_status_received(window, qtbot):
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('torque anomaly')
    assert window.fault_banner.isVisible()

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'PROTECTIVE_STOP', False)
    assert window.fault_banner.isVisible()
    assert 'torque anomaly' in window.fault_banner.text()

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL', False)
    assert not window.fault_banner.isVisible()


@pytest.mark.parametrize('prefix,expected_color', [
    ('PROTECTIVE_STOP: torque anomaly', '#ffb454'),
    ('EMERGENCY_STOP: e-stop pressed', '#ff4d5e'),
    ('FAULT: unexpected force', '#ff4d5e'),
    ('torque anomaly (no prefix)', '#ff4d5e'),
])
def test_fault_after_normal_status_never_shows_green(window, qtbot, prefix, expected_color):
    # 직전 safety_state가 NORMAL이었더라도, Fault 메시지 도착 시 배너가 초록으로
    # 표시되면 안 된다 - 접두어(또는 접두어 없음)에 따른 색만 사용해야 한다.
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL', False)
    assert not window.fault_banner.isVisible()

    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault(prefix)

    assert window.fault_banner.isVisible()
    assert '#39e991' not in window.fault_banner.styleSheet()  # 초록(NORMAL)으로 표시되지 않는다
    assert expected_color in window.fault_banner.styleSheet()


def test_stale_normal_status_right_after_fault_does_not_hide_banner(window, qtbot):
    # Fault 메시지만 도착하고 아직 /task/status가 비정상 상태를 확인해주지 않은
    # 상태에서, (통신 지연 등으로) 과거의 낡은 NORMAL 메시지가 뒤늦게 도착해도
    # 배너를 곧바로 숨기면 안 된다.
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('FAULT: unexpected force')
    assert window.fault_banner.isVisible()

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL', False)

    assert window.fault_banner.isVisible()
    assert 'unexpected force' in window.fault_banner.text()

    # /task/status가 실제로 비정상 상태를 확인해준 뒤에야
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'FAULT', False)
    assert window.fault_banner.isVisible()

    # 그 다음 NORMAL이 와야 비로소 배너가 사라진다.
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL', False)
    assert not window.fault_banner.isVisible()


# ---- 그리퍼 상태 ----

def test_gripper_state_updates_label(window, qtbot):
    with qtbot.waitSignal(window.gripper_state_received, timeout=1000):
        window.ros_client.on_gripper_state(29.4, True)

    assert '29.4' in window.gripper_label.text()
    assert 'True' in window.gripper_label.text()


# ---- 음성 인식: 마이크 음량 게이지 + 마지막 명령어 ----

def test_mic_level_updates_progress_bar(window, qtbot):
    with qtbot.waitSignal(window.mic_level_received, timeout=1000):
        window.ros_client.on_mic_level(0.75)

    assert window.mic_level_bar.value() == 75


def test_mic_level_clips_out_of_range_values(window, qtbot):
    with qtbot.waitSignal(window.mic_level_received, timeout=1000):
        window.ros_client.on_mic_level(1.5)

    assert window.mic_level_bar.value() == 100


def test_stt_command_updates_label(window, qtbot):
    with qtbot.waitSignal(window.stt_command_received, timeout=1000):
        window.ros_client.on_stt_command('스패너 갖다줘')

    assert '스패너 갖다줘' in window.stt_command_label.text()


def test_stt_status_updates_label(window, qtbot):
    with qtbot.waitSignal(window.stt_status_received, timeout=1000):
        window.ros_client.on_stt_status(
            'wakeword_detected', 'wakeWord가 인식되었습니다. 명령어를 말해주세요.', {})

    assert '명령어를 말해주세요' in window.stt_status_label.text()


def test_debug_event_button_collects_warn_events(window, qtbot):
    payload = {
        'node': 'vision_node',
        'level': 'WARN',
        'category': 'TRACK_TOOL',
        'reason': 'target_missing',
        'message': '요청한 tool_class의 유효 3D 추적 결과가 없습니다.',
        'data': {'looking_for': 'wrench'},
    }

    with qtbot.waitSignal(window.debug_event_received, timeout=1000):
        window.ros_client.on_debug_event(payload)

    assert window.debug_log_view.count() == 1
    assert window.debug_toggle_button.text() == '오류 확인 (1)'

    qtbot.mouseClick(window.debug_toggle_button, Qt.LeftButton)

    assert window.debug_panel.isVisible()
    assert window.debug_toggle_button.text() == '오류 확인'


# ---- 연결 상태 ----

def test_connection_status_changes_update_label_and_log(window, qtbot):
    with qtbot.waitSignal(window.connection_changed, timeout=1000):
        window.ros_client.on_connection_changed(True)
    assert '연결됨' in window.connection_label.text()

    with qtbot.waitSignal(window.connection_changed, timeout=1000):
        window.ros_client.on_connection_changed(False)
    assert '연결 안' in window.connection_label.text()


# ---- 카메라: CompressedImage 전달 + Qt Signal ----

def test_camera_image_updates_pixmap(window, qtbot):
    with qtbot.waitSignal(window.camera_image_received, timeout=1000):
        window.ros_client.on_camera_image(_png_bytes())

    assert window.camera_label.pixmap() is not None
    assert not window.camera_label.pixmap().isNull()


def test_camera_image_from_background_thread_updates_ui_via_signal(window, qtbot):
    image_bytes = _png_bytes()

    def emit_from_worker_thread():
        window.ros_client.on_camera_image(image_bytes)

    with qtbot.waitSignal(window.camera_image_received, timeout=2000):
        thread = threading.Thread(target=emit_from_worker_thread)
        thread.start()
        thread.join()

    assert not window.camera_label.pixmap().isNull()


def test_ui_works_without_camera_image(window):
    # Vision 토픽이 아직 없어도 UI는 정상적으로 뜬다.
    assert window.camera_label.text() == '카메라 대기 중...'


# ---- 카메라 영상 끊김 표시/복구 ----

def test_camera_shows_stale_message_after_timeout(window, qtbot):
    with qtbot.waitSignal(window.camera_image_received, timeout=1000):
        window.ros_client.on_camera_image(_png_bytes())
    assert not window.camera_label.pixmap().isNull()

    window.camera_stale_timeout_s = 0.01
    window._last_camera_image_at -= 1.0  # 시간이 흐른 것처럼 시뮬레이션

    window._check_camera_stale()

    assert window._camera_stale is True
    assert window.camera_label.text() == '카메라 영상이 멈췄습니다.'


def test_camera_stale_check_does_not_fire_before_any_image_received(window):
    # 아직 한 번도 영상을 받지 못한 상태에서는 "대기 중..." 문구를 그대로 유지한다.
    window.camera_stale_timeout_s = 0.01

    window._check_camera_stale()

    assert window._camera_stale is False
    assert window.camera_label.text() == '카메라 대기 중...'


def test_camera_recovers_automatically_after_new_image(window, qtbot):
    window.camera_stale_timeout_s = 0.01
    with qtbot.waitSignal(window.camera_image_received, timeout=1000):
        window.ros_client.on_camera_image(_png_bytes())
    window._last_camera_image_at -= 1.0
    window._check_camera_stale()
    assert window._camera_stale is True

    with qtbot.waitSignal(window.camera_image_received, timeout=1000):
        window.ros_client.on_camera_image(_png_bytes())

    assert window._camera_stale is False
    assert not window.camera_label.pixmap().isNull()


# ---- 카메라 오버레이 (영상 안쪽 우측 상단, 로봇 현재 상태) ----

def test_camera_overlay_shows_dash_before_first_status(window):
    assert window.camera_overlay_label.text() == '-'


def test_camera_overlay_shows_translated_state_label(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('DETECT_TRACK', '', 'AUTO', 'NORMAL', False)

    assert window.camera_overlay_label.text() == '탐지 중'


def test_camera_overlay_falls_back_to_raw_state_when_unmapped(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('SOME_NEW_STATE', '', 'AUTO', 'NORMAL', False)

    assert window.camera_overlay_label.text() == 'SOME_NEW_STATE'


def test_camera_overlay_prioritizes_safety_over_task_state(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'FAULT', False)

    assert window.camera_overlay_label.text() == '오류'


def test_camera_overlay_stays_visible_when_camera_is_stale(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL', False)
    window.camera_stale_timeout_s = 0.01
    window._last_camera_image_at = 0.0

    window._check_camera_stale()

    assert window._camera_stale is True
    assert window.camera_overlay_label.text() == '대기중'
    assert window.camera_overlay_label.isVisible()


def test_camera_overlay_stays_top_right_after_resize(window):
    window.resize(700, 500)
    expected_x = window.camera_label.width() - window.camera_overlay_label.width() - 8
    assert window.camera_overlay_label.x() == max(expected_x, 8)
    assert window.camera_overlay_label.y() == 8


def test_camera_stale_timeout_from_env_var(monkeypatch, qtbot):
    monkeypatch.setenv('OPERATOR_GUI_CAMERA_STALE_TIMEOUT_S', '5.5')
    ros_client = _FakeRosClient()
    win = MainWindow(ros_client)
    qtbot.addWidget(win)

    assert win.camera_stale_timeout_s == 5.5


def test_camera_stale_timeout_constructor_arg_overrides_env(monkeypatch, qtbot):
    monkeypatch.setenv('OPERATOR_GUI_CAMERA_STALE_TIMEOUT_S', '5.5')
    ros_client = _FakeRosClient()
    win = MainWindow(ros_client, camera_stale_timeout_s=0.3)
    qtbot.addWidget(win)

    assert win.camera_stale_timeout_s == 0.3


# ---- camera_stale_timeout_s 잘못된 입력값 방어 (finite 양수만 허용) ----

@pytest.mark.parametrize('bad_env_value', ['not-a-number', 'nan', 'inf', '0', '-1.0'])
def test_camera_stale_timeout_falls_back_to_default_on_invalid_env_var(
        monkeypatch, qtbot, bad_env_value):
    monkeypatch.setenv('OPERATOR_GUI_CAMERA_STALE_TIMEOUT_S', bad_env_value)
    ros_client = _FakeRosClient()

    win = MainWindow(ros_client)  # 잘못된 환경변수 하나 때문에 죽지 않아야 한다
    qtbot.addWidget(win)

    assert win.camera_stale_timeout_s == DEFAULT_CAMERA_STALE_TIMEOUT_S


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), 0.0, -1.0])
def test_camera_stale_timeout_falls_back_to_default_on_invalid_constructor_arg(
        qtbot, bad_value):
    ros_client = _FakeRosClient()

    win = MainWindow(ros_client, camera_stale_timeout_s=bad_value)
    qtbot.addWidget(win)

    assert win.camera_stale_timeout_s == DEFAULT_CAMERA_STALE_TIMEOUT_S


def test_camera_stale_timeout_accepts_valid_positive_value(qtbot):
    ros_client = _FakeRosClient()

    win = MainWindow(ros_client, camera_stale_timeout_s=1.5)
    qtbot.addWidget(win)

    assert win.camera_stale_timeout_s == 1.5


# ---- 상태에 따른 버튼 활성화 정책 (UI 편의 기능) ----

def test_move_and_mode_buttons_disabled_before_initial_status(qtbot):
    ros_client = _FakeRosClient()
    win = MainWindow(ros_client)
    qtbot.addWidget(win)
    win.show()

    for attr in ('auto_button', 'manual_button', 'home_button', 'front_button',
                 'up_button', 'down_button', 'watch_button'):
        assert getattr(win, attr).isEnabled() is False, attr
    assert win.stop_button.isEnabled() is True
    assert win.reset_button.isEnabled() is True


@pytest.mark.parametrize('safety_state', [
    'PROTECTIVE_STOP', 'EMERGENCY_STOP', 'FAULT', 'RECOVERY_REQUIRED',
])
def test_move_and_mode_buttons_disabled_when_not_normal(window, qtbot, safety_state):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', safety_state, False)

    for attr in ('auto_button', 'manual_button', 'home_button', 'front_button',
                 'up_button', 'down_button', 'watch_button'):
        assert getattr(window, attr).isEnabled() is False, attr
    # 복구 요청(RESET)과 작업 중단(STOP)은 비정상 상태에서도 항상 눌러야 한다.
    assert window.stop_button.isEnabled() is True
    assert window.reset_button.isEnabled() is True


def test_pose_buttons_enabled_only_in_manual_mode(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL', False)

    for attr in ('home_button', 'front_button', 'up_button', 'down_button', 'watch_button'):
        assert getattr(window, attr).isEnabled() is False, attr

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL', False)

    for attr in ('home_button', 'front_button', 'up_button', 'down_button', 'watch_button'):
        assert getattr(window, attr).isEnabled() is True, attr


def test_move_and_mode_buttons_disabled_while_action_in_progress(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('MOVE_TO_WATCH', '', 'MANUAL', 'NORMAL', False)

    for attr in ('auto_button', 'manual_button', 'home_button', 'front_button',
                 'up_button', 'down_button', 'watch_button'):
        assert getattr(window, attr).isEnabled() is False, attr
    assert window.stop_button.isEnabled() is True
    assert window.reset_button.isEnabled() is True


def test_stop_and_reset_buttons_always_enabled_regardless_of_state(window, qtbot):
    # 작업 중단(STOP)과 복구 요청(RESET)은 서로 다른 목적이지만 둘 다 버튼 정책과
    # 무관하게 항상 눌러야 한다.
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('SERVO_PICK', '', 'AUTO', 'FAULT', False)

    assert window.stop_button.isEnabled() is True
    assert window.reset_button.isEnabled() is True


# ---- 재개(resume) 버튼 ----

def test_resume_button_disabled_before_first_status(window):
    assert window.resume_button.isEnabled() is False


def test_resume_button_disabled_when_not_resumable(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL', False)

    assert window.resume_button.isEnabled() is False


def test_resume_button_enabled_when_task_manager_reports_resumable(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL', True)

    assert window.resume_button.isEnabled() is True


def test_resume_button_click_publishes_resume_command(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL', True)

    qtbot.mouseClick(window.resume_button, Qt.LeftButton)

    assert window.ros_client.published == [CMD_RESUME]


# ---- 종료 시 정리 ----

def test_close_event_closes_ros_client(window):
    window.close()

    assert window.ros_client.closed is True
