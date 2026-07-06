import ctypes

import pytest

from robot_control.drfl_force_monitor import DrflForceMonitor


class _FakeLib:
    """실제 libdsr_hardware2.so 없이 DrflForceMonitor의 판정 로직만 검증하기 위한
    가짜 라이브러리. DrflForceMonitor.__init__이 각 함수에 .restype/.argtypes를
    대입하는데, bound method는 이런 임의 속성 대입을 지원하지 않으므로(매번 새로
    생성되는 객체라 대입이 남지 않는다) 각 "함수"를 인스턴스 속성(클로저)으로
    준비한다 - 일반 함수 객체는 임의 속성 대입이 가능하다."""

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

        def get_external_torque(ctrl):
            return None  # process_sample을 직접 호출하는 테스트에서는 쓰이지 않는다.

        self._CreateRobotControl = create_robot_control
        self._DestroyRobotControl = destroy_robot_control
        self._open_connection = open_connection
        self._close_connection = close_connection
        self._get_external_torque = get_external_torque


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: _FakeLib())
    triggered_calls = []
    m = DrflForceMonitor(
        lib_path='/fake/path.so', robot_ip='127.0.0.1', robot_port=12345,
        thresholds_nm=[5.0, 5.0, 3.0, 1.5, 1.5, 1.5],
        on_triggered=lambda i, v, t: triggered_calls.append((i, v, t)),
        poll_hz=100.0, reset_below_count=3)
    m.triggered_calls = triggered_calls
    return m


def test_open_connection_failure_raises(monkeypatch):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: _FakeLib(open_connection_result=False))

    with pytest.raises(RuntimeError):
        DrflForceMonitor(
            lib_path='/fake/path.so', robot_ip='127.0.0.1', robot_port=12345,
            thresholds_nm=[1.0] * 6, on_triggered=lambda *a: None)


def test_rejects_wrong_threshold_length(monkeypatch):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: _FakeLib())

    with pytest.raises(ValueError):
        DrflForceMonitor(
            lib_path='/fake/path.so', robot_ip='127.0.0.1', robot_port=12345,
            thresholds_nm=[1.0, 2.0], on_triggered=lambda *a: None)


def test_process_sample_triggers_once_when_exceeded(monitor):
    monitor.process_sample([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])  # J6=20.0 > 1.5
    monitor.process_sample([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])  # 계속 초과 - 재발행 안 함

    assert monitor.triggered_calls == [(5, 20.0, 1.5)]


def test_process_sample_ignores_values_within_threshold(monitor):
    monitor.process_sample([0.0] * 6)

    assert monitor.triggered_calls == []


def test_process_sample_requires_consecutive_low_samples_before_reset(monitor):
    monitor.process_sample([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])  # 트립
    monitor.process_sample([0.0] * 6)  # 미만 1회
    monitor.process_sample([0.0] * 6)  # 미만 2회 (reset_below_count=3, 아직 해제 안 됨)
    # 다시 초과 - below_count는 리셋되지만(연속이 끊김) 이미 triggered라 재발행은 안 함
    monitor.process_sample([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])

    assert monitor.triggered_calls == [(5, 20.0, 1.5)]

    monitor.process_sample([0.0] * 6)  # 미만 1회
    monitor.process_sample([0.0] * 6)  # 미만 2회
    monitor.process_sample([0.0] * 6)  # 미만 3회 연속 -> 해제
    monitor.process_sample([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])  # 새로 트립 -> 재발행

    assert monitor.triggered_calls == [(5, 20.0, 1.5), (5, 20.0, 1.5)]


def test_process_sample_reports_first_joint_that_exceeds(monitor):
    monitor.process_sample([0.0, 6.0, 0.0, 0.0, 0.0, 20.0])  # J1=0(정상), J2=6.0>5.0(기준)

    assert monitor.triggered_calls == [(1, 6.0, 5.0)]


def test_process_sample_ignores_malformed_length(monitor):
    monitor.process_sample([0.0, 0.0])  # 길이 6이 아님 - 조용히 무시

    assert monitor.triggered_calls == []


def test_process_sample_swallows_callback_exception(monitor):
    def _raise(*_args):
        raise RuntimeError('콜백 내부 에러')

    monitor._on_triggered = _raise

    monitor.process_sample([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])  # 예외가 전파되지 않아야 한다


def test_stop_closes_connection(monitor):
    monitor.stop()

    assert monitor._ctrl is None
    assert monitor._lib.close_connection_calls == [1234]
