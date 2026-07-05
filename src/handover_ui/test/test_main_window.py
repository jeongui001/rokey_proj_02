import threading

import pytest
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt

from handover_ui.main_window import MainWindow


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
        window.ros_client.on_task_status('MOVE_TO_WATCH', '감시 자세로 이동', 'AUTO', 'NORMAL')

    assert window.state_label.text() == '상태: MOVE_TO_WATCH'
    assert window.detail_label.text() == '디테일: 감시 자세로 이동'
    assert window.mode_label.text() == '모드: AUTO'
    assert window.safety_label.text() == '안전상태: NORMAL'


def test_task_state_change_is_logged_but_first_report_is_not(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL')
    count_after_first = window.log_view.count()

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('MOVE_TO_WATCH', '', 'AUTO', 'NORMAL')

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

    qtbot.mouseClick(win.home_button, Qt.LeftButton)

    assert ros_client.published == []
    assert win.log_view.count() >= 1
    assert '실패' in win.log_view.item(win.log_view.count() - 1).text()


# ---- Fault 배너 상태 ----

@pytest.mark.parametrize('safety_state,expected_color', [
    ('NORMAL', '#2e7d32'),
    ('PROTECTIVE_STOP', '#e65100'),
    ('RECOVERY_REQUIRED', '#e65100'),
    ('EMERGENCY_STOP', '#c62828'),
    ('FAULT', '#c62828'),
])
def test_safety_state_label_color(window, qtbot, safety_state, expected_color):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', safety_state)

    assert expected_color in window.safety_label.styleSheet()


def test_fault_banner_shows_message(window, qtbot):
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('torque anomaly')

    assert window.fault_banner.isVisible()
    assert 'torque anomaly' in window.fault_banner.text()


def test_fault_banner_persists_until_normal_status_received(window, qtbot):
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('torque anomaly')
    assert window.fault_banner.isVisible()

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'PROTECTIVE_STOP')
    assert window.fault_banner.isVisible()
    assert 'torque anomaly' in window.fault_banner.text()

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'MANUAL', 'NORMAL')
    assert not window.fault_banner.isVisible()


@pytest.mark.parametrize('prefix,expected_color', [
    ('PROTECTIVE_STOP: torque anomaly', '#e65100'),
    ('EMERGENCY_STOP: e-stop pressed', '#c62828'),
    ('FAULT: unexpected force', '#c62828'),
    ('torque anomaly (no prefix)', '#c62828'),
])
def test_fault_after_normal_status_never_shows_green(window, qtbot, prefix, expected_color):
    # 직전 safety_state가 NORMAL이었더라도, Fault 메시지 도착 시 배너가 초록으로
    # 표시되면 안 된다 - 접두어(또는 접두어 없음)에 따른 색만 사용해야 한다.
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL')
    assert not window.fault_banner.isVisible()

    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault(prefix)

    assert window.fault_banner.isVisible()
    assert '#2e7d32' not in window.fault_banner.styleSheet()  # 초록으로 표시되지 않는다
    assert expected_color in window.fault_banner.styleSheet()


def test_stale_normal_status_right_after_fault_does_not_hide_banner(window, qtbot):
    # Fault 메시지만 도착하고 아직 /task/status가 비정상 상태를 확인해주지 않은
    # 상태에서, (통신 지연 등으로) 과거의 낡은 NORMAL 메시지가 뒤늦게 도착해도
    # 배너를 곧바로 숨기면 안 된다.
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('FAULT: unexpected force')
    assert window.fault_banner.isVisible()

    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL')

    assert window.fault_banner.isVisible()
    assert 'unexpected force' in window.fault_banner.text()

    # /task/status가 실제로 비정상 상태를 확인해준 뒤에야
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'FAULT')
    assert window.fault_banner.isVisible()

    # 그 다음 NORMAL이 와야 비로소 배너가 사라진다.
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('IDLE', '', 'AUTO', 'NORMAL')
    assert not window.fault_banner.isVisible()


# ---- 그리퍼 상태 ----

def test_gripper_state_updates_label(window, qtbot):
    with qtbot.waitSignal(window.gripper_state_received, timeout=1000):
        window.ros_client.on_gripper_state(29.4, True)

    assert '29.4' in window.gripper_label.text()
    assert 'True' in window.gripper_label.text()


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


# ---- 종료 시 정리 ----

def test_close_event_closes_ros_client(window):
    window.close()

    assert window.ros_client.closed is True
