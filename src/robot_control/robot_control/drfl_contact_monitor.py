import ctypes
import threading
import time


class _RobotForce(ctypes.Structure):
    _fields_ = [('_fForce', ctypes.c_float * 6)]


class DrflContactMonitor:
    """servo_pick 하강 중 TCP 기준 힘(tool force)을 고주기로 감지해, 비전 depth
    추정과 무관하게 실제 접촉 순간을 판정하는 독립 쓰레드.

    DrflForceMonitor(관절 토크 기반 충돌 FAULT 감지, drfl_force_monitor.py)와
    같은 방식으로 ROS2 서비스(GetToolForce)를 거치지 않고 libdsr_hardware2.so에
    ctypes로 직접 연결한다 - 서비스 왕복 지연 없이 고주기로 감지해야 접촉
    순간을 늦게 놓치지 않는다. 안전(FAULT) 목적이 아니라 grasp 높이 확정
    목적이라 DrflForceMonitor와 별도 클래스/연결로 분리했다(다른 책임, 다른
    소비자) - DrflForceMonitor는 이 작업에서 건드리지 않는다.

    on_contact(value, threshold)는 이 클래스의 백그라운드 쓰레드에서 직접
    호출된다 - ROS2 executor 쓰레드가 아니므로, 호출하는 쪽이 쓰레드 안전성을
    책임져야 한다(DrflForceMonitor와 동일 계약). 실제 로봇 모션 명령은 이
    콜백에서 직접 내리지 않아야 한다 - speedl을 스트리밍하는 RT 루프가 유일한
    명령 소유자여야 축 경합이 생기지 않는다(플래그만 세팅하고 다음 RT 틱이
    소비하는 방식을 권장).

    기본값이 suspended=True인 것도 DrflForceMonitor(기본 non-suspended, 항상
    감시)와 다른 점 - 이 모니터는 servo_pick DESCENDING 구간에서만 의미가
    있으므로, 그 구간을 여는 쪽이 명시적으로 resume()해야 한다.
    """

    def __init__(self, lib_path, robot_ip, robot_port, force_threshold_n,
                 on_contact, axis_index=2, ref=0, poll_hz=100.0,
                 reset_below_count=20, stop_join_timeout_s=2.0):
        if force_threshold_n <= 0:
            raise ValueError('force_threshold_n은 양수여야 합니다.')
        if poll_hz <= 0:
            raise ValueError('poll_hz는 양수여야 합니다.')

        self._threshold = float(force_threshold_n)
        self._on_contact = on_contact
        self._axis_index = int(axis_index)
        self._ref = int(ref)
        self._interval = 1.0 / float(poll_hz)
        self._reset_below_count = max(1, int(reset_below_count))
        self._stop_join_timeout_s = float(stop_join_timeout_s)
        self._stop_event = threading.Event()
        self._thread = None
        self._ctrl = None
        self._suspended = True

        lib = ctypes.CDLL(lib_path)
        lib._CreateRobotControl.restype = ctypes.c_void_p
        lib._CreateRobotControl.argtypes = []
        lib._DestroyRobotControl.restype = None
        lib._DestroyRobotControl.argtypes = [ctypes.c_void_p]
        lib._open_connection.restype = ctypes.c_bool
        lib._open_connection.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        lib._close_connection.restype = ctypes.c_bool
        lib._close_connection.argtypes = [ctypes.c_void_p]
        lib._get_tool_force.restype = ctypes.POINTER(_RobotForce)
        lib._get_tool_force.argtypes = [ctypes.c_void_p, ctypes.c_int]
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
        """servo_pick DESCENDING 구간이 아닐 때 on_contact 호출을 막는다. 임계값
        판정/히스테리시스 상태는 평소대로 갱신되므로, resume() 이후에도 리셋
        조건은 그대로 적용된다(DrflForceMonitor와 동일 계약)."""
        self._suspended = True

    def resume(self):
        self._suspended = False

    def _poll_loop(self):
        self._triggered = False
        self._below_count = 0
        while not self._stop_event.is_set():
            time.sleep(self._interval)
            try:
                ptr = self._lib._get_tool_force(self._ctrl, self._ref)
            except Exception:
                continue
            if not ptr:
                continue
            self.process_sample(list(ptr.contents._fForce))

    def process_sample(self, force):
        """force(6성분: fx,fy,fz,mx,my,mz)에서 axis_index 성분만 임계값과
        비교해 히스테리시스 상태를 갱신하고, 새로 트립됐으면 on_contact
        콜백을 호출한다. _poll_loop에서 실제 읽은 값으로 호출되며, 쓰레드/
        타이밍 없이 테스트하기 위해 별도 메서드로 분리했다(DrflForceMonitor와
        동일 패턴)."""
        if len(force) != 6:
            return
        if not hasattr(self, '_triggered'):
            self._triggered = False
            self._below_count = 0

        value = force[self._axis_index]
        exceeded = abs(value) > self._threshold

        if exceeded:
            self._below_count = 0
            if not self._triggered:
                self._triggered = True
                if not self._suspended:
                    try:
                        self._on_contact(value, self._threshold)
                    except Exception:
                        pass
        elif self._triggered:
            self._below_count += 1
            if self._below_count >= self._reset_below_count:
                self._triggered = False
                self._below_count = 0
