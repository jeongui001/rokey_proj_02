import ctypes
import threading
import time


class _RobotForce(ctypes.Structure):
    _fields_ = [('_fForce', ctypes.c_float * 6)]


class DrflForceMonitor:
    """dsr_hardware2의 DRFL 라이브러리(libdsr_hardware2.so)에 ctypes로 직접 연결해
    외력을 고주기(기본 100Hz)로 감지하는 독립 쓰레드.

    ROS2 executor/서비스(dsr_msgs2 GetExternalTorque)를 거치지 않기 때문에, 로봇이
    MOVING 중이어도 실시간으로 감지할 수 있다 - rokey_proj_01의 force_monitor_node.py와
    동일한 접근(2026-07-06 도입). SafetyMonitor의 STANDBY 전용 delta 체크는 정지
    상태에서의 미세한 접촉 감지용으로 그대로 두고, 이 모니터는 이동 중을 포함한
    전체 시간에 대해 관절별 절대 임계값 + 히스테리시스로 감지하는 별도 레이어다.

    on_triggered(joint_index, value, threshold_nm)는 이 클래스의 백그라운드
    쓰레드에서 직접 호출된다 - ROS2 executor 쓰레드가 아니므로, 호출하는 쪽이
    쓰레드 안전성을 책임져야 한다.
    """

    def __init__(self, lib_path, robot_ip, robot_port, thresholds_nm,
                 on_triggered, poll_hz=100.0, reset_below_count=20,
                 stop_join_timeout_s=2.0):
        if len(thresholds_nm) != 6:
            raise ValueError('thresholds_nm은 관절 6개 값이어야 합니다.')
        if poll_hz <= 0:
            raise ValueError('poll_hz는 양수여야 합니다.')

        self._thresholds = [float(v) for v in thresholds_nm]
        self._on_triggered = on_triggered
        self._interval = 1.0 / float(poll_hz)
        self._reset_below_count = max(1, int(reset_below_count))
        self._stop_join_timeout_s = float(stop_join_timeout_s)
        self._stop_event = threading.Event()
        self._thread = None
        self._ctrl = None
        self._suspended = False

        lib = ctypes.CDLL(lib_path)
        lib._CreateRobotControl.restype = ctypes.c_void_p
        lib._CreateRobotControl.argtypes = []
        lib._DestroyRobotControl.restype = None
        lib._DestroyRobotControl.argtypes = [ctypes.c_void_p]
        lib._open_connection.restype = ctypes.c_bool
        lib._open_connection.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        lib._close_connection.restype = ctypes.c_bool
        lib._close_connection.argtypes = [ctypes.c_void_p]
        lib._get_external_torque.restype = ctypes.POINTER(_RobotForce)
        lib._get_external_torque.argtypes = [ctypes.c_void_p]
        self._lib = lib

        ctrl = lib._CreateRobotControl()
        if not lib._open_connection(ctrl, str(robot_ip).encode(), int(robot_port)):
            lib._DestroyRobotControl(ctrl)
            raise RuntimeError(f'DRFL 직접 연결 실패: {robot_ip}:{robot_port}')
        self._ctrl = ctrl

    def start(self):
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._stop_join_timeout_s)
            self._thread = None
        if self._ctrl is not None:
            try:
                self._lib._close_connection(self._ctrl)
                self._lib._DestroyRobotControl(self._ctrl)
            except Exception:
                pass
            self._ctrl = None

    def suspend(self):
        """handover_hold의 컴플라이언스 구간처럼 사람의 접촉력이 기대되는 동안
        on_triggered 호출을 일시적으로 막는다. 임계값 판정/히스테리시스 상태는
        평소대로 갱신되므로, resume() 이후에도 리셋 조건은 그대로 적용된다."""
        self._suspended = True

    def resume(self):
        self._suspended = False

    def _poll_loop(self):
        self._triggered = False
        self._below_count = 0
        while not self._stop_event.is_set():
            time.sleep(self._interval)
            try:
                ptr = self._lib._get_external_torque(self._ctrl)
            except Exception:
                continue
            if not ptr:
                continue
            self.process_sample(list(ptr.contents._fForce))

    def process_sample(self, torque):
        """torque(관절 6개)를 관절별 임계값과 비교해 히스테리시스 상태를 갱신하고,
        새로 트립됐으면 on_triggered 콜백을 호출한다. _poll_loop에서 실제 읽은 값으로
        호출되며, 쓰레드/타이밍 없이 테스트하기 위해 별도 메서드로 분리했다."""
        if len(torque) != 6:
            return
        if not hasattr(self, '_triggered'):
            self._triggered = False
            self._below_count = 0

        exceeded = None
        for index, (value, threshold) in enumerate(zip(torque, self._thresholds)):
            if abs(value) > threshold:
                exceeded = (index, value, threshold)
                break

        if exceeded is not None:
            self._below_count = 0
            if not self._triggered:
                self._triggered = True
                if not self._suspended:
                    index, value, threshold = exceeded
                    try:
                        self._on_triggered(index, value, threshold)
                    except Exception:
                        pass
        elif self._triggered:
            self._below_count += 1
            if self._below_count >= self._reset_below_count:
                self._triggered = False
                self._below_count = 0
