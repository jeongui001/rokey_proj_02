import logging
import threading
import time

_LOGGER = logging.getLogger(__name__)


class SpeedlWatchdog:
    """메인 서보 루프와 별개 스레드에서 도는 데드맨스위치.

    루프가 pet()을 timeout_s 이내에 호출하지 않으면(예외로 루프가 죽거나 행이
    걸린 경우) 워치독 스레드가 독립적으로 on_timeout()을 호출한다. rclpy에
    의존하지 않는 순수 파이썬 클래스라 하드웨어 없이 유닛 테스트 가능하다.

    비-RT speedl은 명령이 끊겨도 스스로 멈추지 않지만(2026-07-07
    probe_speedl_stream.py 실측 확인), vel=0을 발행하면 로봇이 멈춘 채
    유지된다(같은 실측). 타임아웃이 지속되는 동안은 poll_interval_s마다
    on_timeout을 계속 호출한다(스레드가 죽지 않고 감시를 유지) - 이후 pet()이
    다시 들어와 타임아웃 조건이 해소되면 자동으로 재무장되어 추가 호출 없이
    감시만 계속한다. 즉 "한 번 쏘고 죽는" 1회성 알람이 아니라, 루프가 실제로
    다시 살아날 때까지 정지를 계속 보장하는 지속적 데드맨스위치다.

    같은 프로세스 내 스레드 기반이라 메인 루프가 행(hang)에 걸리거나 예외로
    죽어도 동작하지만, 프로세스 자체가 죽는 경우(kill -9, segfault)는 보호
    범위 밖이다.
    """

    def __init__(self, timeout_s, on_timeout, poll_interval_s=0.05):
        self._timeout_s = timeout_s
        self._on_timeout = on_timeout
        self._poll_interval_s = poll_interval_s
        self._last_pet = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._last_pet = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pet(self) -> None:
        self._last_pet = time.monotonic()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_s + self._poll_interval_s + 1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if time.monotonic() - self._last_pet > self._timeout_s:
                try:
                    self._on_timeout()
                except Exception:
                    _LOGGER.exception('SpeedlWatchdog on_timeout 콜백 중 예외 발생')
            time.sleep(self._poll_interval_s)


__all__ = ['SpeedlWatchdog']
