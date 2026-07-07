import time

from robot_control.speedl_watchdog import SpeedlWatchdog


def test_pet_prevents_timeout():
    calls = []
    wd = SpeedlWatchdog(
        timeout_s=0.1, on_timeout=lambda: calls.append(True), poll_interval_s=0.02)
    wd.start()
    try:
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            wd.pet()
            time.sleep(0.02)
    finally:
        wd.stop()
    assert calls == []


def test_timeout_fires_once_when_pet_stops():
    calls = []
    wd = SpeedlWatchdog(
        timeout_s=0.05, on_timeout=lambda: calls.append(True), poll_interval_s=0.01)
    wd.start()
    time.sleep(0.2)
    wd.stop()
    assert calls == [True]


def test_stop_prevents_timeout_after_pet_stops():
    calls = []
    wd = SpeedlWatchdog(
        timeout_s=0.2, on_timeout=lambda: calls.append(True), poll_interval_s=0.02)
    wd.start()
    wd.stop()
    time.sleep(0.3)
    assert calls == []


def test_stop_without_start_does_not_raise():
    wd = SpeedlWatchdog(timeout_s=0.1, on_timeout=lambda: None)
    wd.stop()  # start() 없이 stop()만 호출해도 예외 없이 안전해야 한다
