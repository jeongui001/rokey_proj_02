import inspect
import math
import time

from robot_control.safety_monitor import SafetyState


def _import_modbus_tcp_client():
    """pymodbus는 3.0에서 import 경로를 pymodbus.client.sync.ModbusTcpClient에서
    pymodbus.client.ModbusTcpClient로 옮겼다. 팀원마다 설치된 pymodbus 버전이
    다를 수 있어(일부는 2.x대가 남아있는 환경) 신버전 경로를 먼저 시도하고
    실패하면 구버전 경로로 폴백한다."""
    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError:
        from pymodbus.client.sync import ModbusTcpClient
    return ModbusTcpClient


def _resolve_modbus_id_kwarg(client) -> str:
    """pymodbus는 버전마다 슬레이브/디바이스 ID를 넘기는 키워드 인자 이름이
    하위호환 없이 완전히 바뀌어왔다: 2.x는 unit=, 3.0~3.11은 slave=, 3.12+는
    device_id=. 팀원마다 pip이 설치해준 pymodbus 버전이 다를 수 있으므로(같은
    Jazzy에서도 설치 시점에 따라 3.6대/3.13대가 섞여 있을 수 있음) 하드코딩 대신
    실제 설치된 클라이언트 클래스의 시그니처를 보고 맞는 이름을 고른다."""
    params = inspect.signature(type(client).read_holding_registers).parameters
    for name in ('device_id', 'slave', 'unit'):
        if name in params:
            return name
    # 세 이름 모두 시그니처에 없다는 건 read_holding_registers가 **kwargs로만
    # 받는다는 뜻이다 - 이건 pymodbus 2.x의 실제 시그니처(read_holding_registers(self,
    # address, count=1, **kwargs), unit_id는 ModbusPDU.__init__이 내부적으로
    # kwargs.get('unit', ...)으로 읽음)이며, 3.0 이후 버전은 계속 이름 있는
    # 키워드 인자로 명시해왔으므로 unit을 기본값으로 쓴다. (device_id를 기본값으로
    # 쓰면 2.x에서 unit 인자가 그냥 조용히 무시되어 unit_id가 0으로 통신하게 되고
    # 툴체인저(id=65)와 통신이 안 되어 Modbus 오류가 난다.)
    return 'unit'


class RG2Status:
    SUCCESS = 'SUCCESS'
    CANCELED = 'CANCELED'
    FAULT = 'FAULT'
    TIMEOUT = 'TIMEOUT'
    COMMUNICATION_ERROR = 'COMMUNICATION_ERROR'
    INVALID_INPUT = 'INVALID_INPUT'


class RG2Client:
    """OnRobot RG2/RG6 Modbus TCP 통신 경계."""

    _CMD_STOP = 8
    _CMD_GRIP_W_OFFSET = 16
    _TOOL_CHANGER_MODBUS_ID = 65  # OnRobot 툴체인저의 고정 Modbus 슬레이브/디바이스 ID
    MAX_WIDTH_MM = {'rg2': 110.0, 'rg6': 160.0}
    MAX_FORCE_N = {'rg2': 40.0, 'rg6': 120.0}
    _DEFAULT_COMMAND_TIMEOUT_S = 5.0
    _DEFAULT_POLL_INTERVAL_S = 0.05
    _DEFAULT_OPEN_WIDTH_TOLERANCE_MM = 2.0
    _DEFAULT_CONNECT_TIMEOUT_S = 2.0
    # node 없이(node=None) 생성된 경우의 기본값 - robot_control_node를 통해 쓰일
    # 때는 항상 node가 주어지므로 실제로는 각 파라미터 기본값이 적용된다. 여기
    # 기본값을 0으로 두는 건 node 없는 단위 테스트/스크립트에서 재시도로 인한
    # 예상치 못한 지연을 피하기 위해서다.
    _DEFAULT_COMMUNICATION_RETRY_COUNT = 0
    _DEFAULT_COMMUNICATION_RETRY_BACKOFF_S = 0.5

    def __init__(
            self, ip: str, port: int = 502, hardware_enabled: bool = False,
            gripper: str = 'rg2', node=None):
        self.ip = ip
        self.port = port
        self.hardware_enabled = hardware_enabled
        self.gripper = gripper
        self._node = node
        self._client = None
        self._modbus_id_kwarg = None  # _ensure_connected가 접속 시 감지해 채운다
        self._sim_width_mm = self.MAX_WIDTH_MM.get(gripper, 110.0)
        self._sim_grip_detected = False
        self.last_status = None
        self.last_width_mm = None
        self.last_grip_detected = None

    def _ensure_connected(self):
        if self._client is None:
            ModbusTcpClient = _import_modbus_tcp_client()
            # stopbits/bytesize/parity/baudrate는 시리얼(RTU) 전용 파라미터라 Modbus
            # TCP에는 애초에 의미가 없었다 - 예전 pymodbus는 **kwargs로 조용히
            # 무시했지만, 3.12+는 ModbusTcpClient.__init__에서 **kwargs 자체를
            # 없애 명시하지 않은 키워드 인자를 주면 TypeError가 난다.
            self._client = ModbusTcpClient(
                self.ip, port=self.port, timeout=self._connect_timeout_s())
            self._modbus_id_kwarg = _resolve_modbus_id_kwarg(self._client)
        try:
            if not self._client.connect():
                self._log_comm_failure('connect', detail='connect()가 False를 반환')
                self._reset_connection()
                return None
        except Exception as exc:
            self._log_comm_failure('connect', exc=exc)
            self._reset_connection()
            return None
        return self._client

    def _log_comm_failure(self, context, exc=None, response=None, detail=None):
        # COMMUNICATION_ERROR가 반복될 때 "왜"인지 전부 삼켜져서(2026-07-09: 재연결
        # 로직을 넣었는데도 여전히 재현) 원인을 못 보고 있었다 - 실제 pymodbus
        # 예외/응답 내용을 그대로 남긴다. RG2Client는 유닛 테스트에서 node=None으로도
        # 쓰이므로 get_logger()는 node가 있을 때만 부른다.
        if detail is None:
            if exc is not None:
                detail = f'{type(exc).__name__}: {exc}'
            elif response is not None:
                detail = repr(response)
            else:
                detail = 'unknown'
        if self._node is not None:
            self._node.get_logger().error(
                f'RG2 통신 실패[{context}]: {detail} (ip={self.ip}, port={self.port})')

    def _reset_connection(self):
        # pymodbus의 ModbusTcpClient.connect()는 소켓이 상대(RG2)쪽에서 이미 끊긴
        # "죽은" 상태여도 이미 연결됐다고 착각하고 그냥 통과시킨다 - 그러면 이후
        # read/write가 계속 실패하는데 재시도도 같은 죽은 소켓으로만 이뤄져 절대
        # 회복이 안 된다(2026-07-09 실기: RG2 open이 재시도까지 다 실패하고 FAULT).
        # 통신 실패를 감지한 곳마다 이걸 불러서 다음 시도(재시도 포함)가 새 TCP
        # 연결로 다시 붙게 한다.
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _command_timeout_s(self) -> float:
        if self._node is not None:
            return float(self._node.get_parameter('rg2.command_timeout_s').value)
        return self._DEFAULT_COMMAND_TIMEOUT_S

    def _poll_interval_s(self) -> float:
        if self._node is not None:
            value = float(self._node.get_parameter('rg2.poll_interval_s').value)
            return max(value, 0.001)
        return self._DEFAULT_POLL_INTERVAL_S

    def _open_width_tolerance_mm(self) -> float:
        if self._node is not None:
            return float(self._node.get_parameter('rg2.open_width_tolerance_mm').value)
        return self._DEFAULT_OPEN_WIDTH_TOLERANCE_MM

    def _connect_timeout_s(self) -> float:
        # 이전에는 ModbusTcpClient의 소켓 타임아웃이 1초로 하드코딩돼 있어,
        # 네트워크가 잠깐 느려지기만 해도 COMMUNICATION_ERROR로 이어졌다.
        if self._node is not None:
            return float(self._node.get_parameter('rg2.connect_timeout_s').value)
        return self._DEFAULT_CONNECT_TIMEOUT_S

    def _communication_retry_count(self) -> int:
        if self._node is not None:
            return max(0, int(self._node.get_parameter('rg2.communication_retry_count').value))
        return self._DEFAULT_COMMUNICATION_RETRY_COUNT

    def _communication_retry_backoff_s(self) -> float:
        if self._node is not None:
            return max(
                0.0, float(self._node.get_parameter('rg2.communication_retry_backoff_s').value))
        return self._DEFAULT_COMMUNICATION_RETRY_BACKOFF_S

    def _validate_inputs(self, width_mm: float, force_n: float) -> bool:
        if self.gripper not in self.MAX_WIDTH_MM or self.gripper not in self.MAX_FORCE_N:
            return False
        if not (math.isfinite(width_mm) and math.isfinite(force_n)):
            return False
        return (
            0.0 <= width_mm <= self.MAX_WIDTH_MM[self.gripper]
            and 0.0 <= force_n <= self.MAX_FORCE_N[self.gripper]
        )

    @staticmethod
    def _response_failed(response) -> bool:
        return (
            response is None
            or (hasattr(response, 'isError') and response.isError())
        )

    def _id_kwargs(self) -> dict:
        return {self._modbus_id_kwarg: self._TOOL_CHANGER_MODBUS_ID}

    def _read_busy_bit(self):
        try:
            client = self._ensure_connected()
            if client is None:
                return None
            response = client.read_holding_registers(address=268, count=1, **self._id_kwargs())
            if self._response_failed(response):
                self._log_comm_failure('read_busy_bit', response=response)
                self._reset_connection()
                return None
            registers = getattr(response, 'registers', None)
            return None if not registers else bool(registers[0] & 0x01)
        except Exception as exc:
            self._log_comm_failure('read_busy_bit', exc=exc)
            self._reset_connection()
            return None

    def _read_final_state(self):
        try:
            client = self._ensure_connected()
            if client is None:
                return None
            width_response = client.read_holding_registers(address=267, count=1, **self._id_kwargs())
            status_response = client.read_holding_registers(address=268, count=1, **self._id_kwargs())
            if self._response_failed(width_response) or self._response_failed(status_response):
                self._log_comm_failure(
                    'read_final_state',
                    response=width_response if self._response_failed(width_response)
                    else status_response)
                self._reset_connection()
                return None
            width_registers = getattr(width_response, 'registers', None)
            status_registers = getattr(status_response, 'registers', None)
            if not width_registers or not status_registers:
                return None
            width_mm = width_registers[0] / 10.0
            if not math.isfinite(width_mm):
                return None
            return width_mm, bool(status_registers[0] & 0x02)
        except Exception as exc:
            self._log_comm_failure('read_final_state', exc=exc)
            self._reset_connection()
            return None

    def _wait_until_not_busy(self, goal_handle=None):
        if not self.hardware_enabled:
            return RG2Status.SUCCESS
        deadline = time.monotonic() + self._command_timeout_s()
        while True:
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return RG2Status.CANCELED
            if self._node is not None and self._node.safety_state != SafetyState.NORMAL:
                return RG2Status.FAULT
            busy = self._read_busy_bit()
            if busy is None:
                return RG2Status.COMMUNICATION_ERROR
            if not busy:
                return RG2Status.SUCCESS
            if time.monotonic() >= deadline:
                return RG2Status.TIMEOUT
            time.sleep(self._poll_interval_s())

    def _send_stop_and_wait(self):
        client = self._ensure_connected()
        if client is None:
            return False
        try:
            response = client.write_registers(
                address=0, values=[0, 0, self._CMD_STOP], **self._id_kwargs())
        except Exception as exc:
            self._log_comm_failure('send_stop', exc=exc)
            self._reset_connection()
            return False
        if self._response_failed(response):
            self._log_comm_failure('send_stop', response=response)
            self._reset_connection()
            return False
        deadline = time.monotonic() + self._command_timeout_s()
        while True:
            busy = self._read_busy_bit()
            if busy is None:
                return False
            if not busy:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self._poll_interval_s())

    def _check_before_command(self, goal_handle=None):
        if goal_handle is not None and goal_handle.is_cancel_requested:
            return RG2Status.CANCELED
        if self._node is not None and self._node.safety_state != SafetyState.NORMAL:
            return RG2Status.FAULT
        if not self.hardware_enabled:
            return RG2Status.SUCCESS
        return self._wait_until_not_busy(goal_handle=goal_handle)

    def _run_command(self, goal_handle, values, max_width):
        status = self._check_before_command(goal_handle=goal_handle)
        if status != RG2Status.SUCCESS:
            return status
        client = self._ensure_connected()
        if client is None:
            return RG2Status.COMMUNICATION_ERROR
        try:
            response = client.write_registers(address=0, values=values, **self._id_kwargs())
        except Exception as exc:
            self._log_comm_failure('write_command', exc=exc)
            self._reset_connection()
            return RG2Status.COMMUNICATION_ERROR
        if self._response_failed(response):
            self._log_comm_failure('write_command', response=response)
            self._reset_connection()
            return RG2Status.COMMUNICATION_ERROR

        status = self._wait_until_not_busy(goal_handle=goal_handle)
        if status in (RG2Status.CANCELED, RG2Status.FAULT):
            if not self._send_stop_and_wait():
                return RG2Status.COMMUNICATION_ERROR
            return status
        if status != RG2Status.SUCCESS:
            return status

        final = self._read_final_state()
        if final is None:
            return RG2Status.COMMUNICATION_ERROR
        width_mm, grip_detected = final
        # 캘리브레이션 오차로 실측 최대폭이 스펙값(max_width)을 살짝 넘는 경우까지
        # "통신 오류"로 오판하지 않도록 open_width_tolerance_mm만큼 여유를 둔다
        # (2026-07-11 실기: 그리퍼가 실제로는 다 열렸는데 COMMUNICATION_ERROR로 잘못
        # 분류되어 재시도 끝에 FAULT로 이어짐).
        tolerance = self._open_width_tolerance_mm()
        if width_mm < 0.0 or width_mm > max_width + tolerance:
            return RG2Status.COMMUNICATION_ERROR
        self.last_width_mm = width_mm
        self.last_grip_detected = grip_detected
        return RG2Status.SUCCESS

    def _run_command_with_retry(self, goal_handle, values, max_width):
        """COMMUNICATION_ERROR에 한해서만 자동 재시도한다 - 같은 목표(width/force)를
        그대로 다시 보내는 멱등한 재시도라, 이전 시도가 실제로는 그리퍼에 전달됐어도
        같은 명령을 또 보내는 것뿐이라 안전하다. CANCELED/FAULT/INVALID_INPUT 등
        다른 상태는 재시도하지 않고 즉시 반환한다(취소/안전정지를 재시도로 덮어쓰지
        않기 위함)."""
        retries = self._communication_retry_count()
        backoff_s = self._communication_retry_backoff_s()
        attempt = 0
        while True:
            status = self._run_command(goal_handle, values, max_width)
            if status != RG2Status.COMMUNICATION_ERROR or attempt >= retries:
                return status
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return status
            if self._node is not None and self._node.safety_state != SafetyState.NORMAL:
                return status
            attempt += 1
            if self._node is not None:
                self._node.get_logger().warn(
                    f'RG2 통신 오류 - 재시도 {attempt}/{retries}')
            time.sleep(backoff_s)

    def open(self, goal_handle=None) -> bool:
        self.last_status = None
        self.last_width_mm = None
        self.last_grip_detected = None
        if self.gripper not in self.MAX_WIDTH_MM or self.gripper not in self.MAX_FORCE_N:
            self.last_status = RG2Status.INVALID_INPUT
            if self._node is not None:
                self._node.get_logger().error(f'지원하지 않는 RG2/RG6 gripper 이름입니다: {self.gripper}')
            return False
        max_width = self.MAX_WIDTH_MM[self.gripper]
        max_force = self.MAX_FORCE_N[self.gripper]
        if not self.hardware_enabled:
            self._sim_width_mm = max_width
            self._sim_grip_detected = False
            self.last_status = RG2Status.SUCCESS
            self.last_width_mm = max_width
            self.last_grip_detected = False
            return True

        self.last_status = self._run_command_with_retry(
            goal_handle,
            [int(max_force * 10), int(max_width * 10), self._CMD_GRIP_W_OFFSET],
            max_width)
        if self.last_status != RG2Status.SUCCESS:
            if self._node is not None:
                self._node.get_logger().error(
                    f'RG2 open 명령이 실패했습니다 '
                    f'(status={self.last_status}, gripper={self.gripper}, '
                    f'max_width_mm={max_width}, max_force_n={max_force}).')
            return False
        tolerance = self._open_width_tolerance_mm()
        if self.last_width_mm is not None and self.last_width_mm < max_width - tolerance:
            if self._node is not None:
                self._node.get_logger().warn(
                    f'RG2 open 최종 폭 {self.last_width_mm}mm가 '
                    f'허용오차 {tolerance}mm 밖입니다.')
        return True

    def close(self, width_mm: float, force_n: float, goal_handle=None) -> bool:
        self.last_status = None
        self.last_width_mm = None
        self.last_grip_detected = None
        if not self._validate_inputs(width_mm, force_n):
            self.last_status = RG2Status.INVALID_INPUT
            if self._node is not None:
                self._node.get_logger().error(
                    f'RG2 close width/force 입력값이 범위를 벗어났습니다: '
                    f'gripper={self.gripper}, width_mm={width_mm}, force_n={force_n}, '
                    f'max_width_mm={self.MAX_WIDTH_MM.get(self.gripper)}, '
                    f'max_force_n={self.MAX_FORCE_N.get(self.gripper)}')
            return False
        if not self.hardware_enabled:
            self._sim_width_mm = width_mm
            self._sim_grip_detected = True
            self.last_status = RG2Status.SUCCESS
            self.last_width_mm = width_mm
            self.last_grip_detected = True
            return True

        self.last_status = self._run_command_with_retry(
            goal_handle,
            [int(force_n * 10), int(width_mm * 10), self._CMD_GRIP_W_OFFSET],
            self.MAX_WIDTH_MM[self.gripper])
        if self.last_status != RG2Status.SUCCESS:
            if self._node is not None:
                self._node.get_logger().error(
                    f'RG2 close 명령이 실패했습니다 '
                    f'(status={self.last_status}, gripper={self.gripper}, '
                    f'width_mm={width_mm}, force_n={force_n}).')
        return self.last_status == RG2Status.SUCCESS

    def get_state(self):
        if not self.hardware_enabled:
            return self._sim_width_mm, self._sim_grip_detected
        return self._read_final_state() or (0.0, False)


__all__ = ['RG2Client', 'RG2Status']
