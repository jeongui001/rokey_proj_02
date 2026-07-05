import pytest
from PyQt5.QtCore import Qt

from handover_ui.main_window import MainWindow


class _FakeRosClient:
    def __init__(self):
        self.published = []
        self.on_task_status = None
        self.on_gripper_state = None
        self.on_fault = None

    def publish_command(self, text):
        self.published.append(text)


@pytest.fixture
def window(qtbot):
    ros_client = _FakeRosClient()
    win = MainWindow(ros_client)
    qtbot.addWidget(win)
    win.show()
    return win


def test_task_status_updates_labels(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('MOVE_TO_WATCH', '감시 자세로 이동')

    assert window.state_label.text() == '상태: MOVE_TO_WATCH'
    assert window.detail_label.text() == '디테일: 감시 자세로 이동'
    assert window.log_view.count() == 1


def test_gripper_state_updates_label(window, qtbot):
    with qtbot.waitSignal(window.gripper_state_received, timeout=1000):
        window.ros_client.on_gripper_state(29.4, True)

    assert '29.4' in window.gripper_label.text()
    assert 'True' in window.gripper_label.text()


def test_fault_shows_banner(window, qtbot):
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('torque anomaly')

    assert window.fault_banner.isVisible()
    assert 'torque anomaly' in window.fault_banner.text()


def test_send_button_publishes_command_and_clears_input(window, qtbot):
    window.command_input.setText('스패너 갖다줘')
    qtbot.mouseClick(window.send_button, Qt.LeftButton)

    assert window.ros_client.published == ['스패너 갖다줘']
    assert window.command_input.text() == ''


def test_send_button_ignores_empty_input(window, qtbot):
    window.command_input.setText('   ')
    qtbot.mouseClick(window.send_button, Qt.LeftButton)

    assert window.ros_client.published == []
