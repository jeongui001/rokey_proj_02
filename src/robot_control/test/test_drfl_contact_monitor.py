import ctypes

import pytest

from robot_control.drfl_contact_monitor import DrflContactMonitor


class _FakeLib:
    """실제 libdsr_hardware2.so 없이 DrflContactMonitor의 판정 로직만 검증하기
    위한 가짜 라이브러리. test_drfl_force_monitor.py의 _FakeLib과 동일한 이유로
    각 "함수"를 인스턴스 속성(클로저)으로 준비한다(bound method는 임의 속성
    대입을 지원하지 않는다)."""

    def __init__(self, open_connection_result=True):
        self.close_connection_calls = []

        def create_robot_control():
            return 1234

        def destroy_robot_control(ctrl):
            pass

        def open_connection(ctrl, ip, port):
            return open_connection_result

        def close_connection(ctrl):
            self.close_connection_calls.append(ctrl)
            return True

        def get_tool_force(ctrl, ref):
            return None  # process_sample을 직접 호출하는 테스트에서는 쓰이지 않는다.

        self._CreateRobotControl = create_robot_control
        self._DestroyRobotControl = destroy_robot_control
        self._open_connection = open_connection
        self._close_connection = close_connection
        self._get_tool_force = get_tool_force


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: _FakeLib())
    contact_calls = []
    m = DrflContactMonitor(
        lib_path='/fake/path.so', robot_ip='127.0.0.1', robot_port=12345,
        force_threshold_n=5.0,
        on_contact=lambda value, threshold: contact_calls.append((value, threshold)),
        poll_hz=100.0, reset_below_count=3)
    m.contact_calls = contact_calls
    m.resume()  # 기본은 suspended=True(안전 기본값) - 판정 로직 테스트는 armed 상태에서 확인
    return m


def test_open_connection_failure_raises(monkeypatch):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: _FakeLib(open_connection_result=False))

    with pytest.raises(RuntimeError):
        DrflContactMonitor(
            lib_path='/fake/path.so', robot_ip='127.0.0.1', robot_port=12345,
            force_threshold_n=5.0, on_contact=lambda *a: None)


def test_rejects_non_positive_threshold(monkeypatch):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: _FakeLib())

    with pytest.raises(ValueError):
        DrflContactMonitor(
            lib_path='/fake/path.so', robot_ip='127.0.0.1', robot_port=12345,
            force_threshold_n=0.0, on_contact=lambda *a: None)


def test_starts_suspended_by_default(monkeypatch):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: _FakeLib())
    contact_calls = []
    m = DrflContactMonitor(
        lib_path='/fake/path.so', robot_ip='127.0.0.1', robot_port=12345,
        force_threshold_n=5.0,
        on_contact=lambda value, threshold: contact_calls.append((value, threshold)))

    m.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # fz=20.0 > 5.0이지만 resume() 전

    assert contact_calls == []


def test_process_sample_triggers_once_when_exceeded(monitor):
    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # fz=20.0 > 5.0
    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # 계속 초과 - 재발행 안 함

    assert monitor.contact_calls == [(20.0, 5.0)]


def test_process_sample_ignores_values_within_threshold(monitor):
    monitor.process_sample([0.0] * 6)

    assert monitor.contact_calls == []


def test_process_sample_requires_consecutive_low_samples_before_reset(monitor):
    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # 트립
    monitor.process_sample([0.0] * 6)  # 미만 1회
    monitor.process_sample([0.0] * 6)  # 미만 2회 (reset_below_count=3, 아직 해제 안 됨)
    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # 다시 초과 - 이미 triggered라 재발행 안 함

    assert monitor.contact_calls == [(20.0, 5.0)]

    monitor.process_sample([0.0] * 6)  # 미만 1회
    monitor.process_sample([0.0] * 6)  # 미만 2회
    monitor.process_sample([0.0] * 6)  # 미만 3회 연속 -> 해제
    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # 새로 트립 -> 재발행

    assert monitor.contact_calls == [(20.0, 5.0), (20.0, 5.0)]


def test_process_sample_ignores_malformed_length(monitor):
    monitor.process_sample([0.0, 0.0])  # 길이 6이 아님 - 조용히 무시

    assert monitor.contact_calls == []


def test_process_sample_swallows_callback_exception(monitor):
    def _raise(*_args):
        raise RuntimeError('콜백 내부 에러')

    monitor._on_contact = _raise

    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # 예외가 전파되지 않아야 한다


def test_stop_closes_connection(monitor):
    monitor.stop()

    assert monitor._ctrl is None
    assert monitor._lib.close_connection_calls == [1234]


def test_process_sample_does_not_trigger_while_suspended(monitor):
    monitor.suspend()
    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])

    assert monitor.contact_calls == []


def test_process_sample_triggers_normally_after_resume(monitor):
    monitor.suspend()
    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # 억제됨 - 콜백 없음
    monitor.process_sample([0.0] * 6)  # 미만 1회
    monitor.process_sample([0.0] * 6)  # 미만 2회
    monitor.process_sample([0.0] * 6)  # 미만 3회 연속 -> latch 해제(reset_below_count=3)
    monitor.resume()
    monitor.process_sample([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # 재개 후 초과 -> 정상 발행

    assert monitor.contact_calls == [(20.0, 5.0)]
