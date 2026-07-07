import math
import time

from robot_control.safety_monitor import SafetyState


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
    MAX_WIDTH_MM = {'rg2': 110.0, 'rg6': 160.0}
    MAX_FORCE_N = {'rg2': 40.0, 'rg6': 120.0}
    _DEFAULT_COMMAND_TIMEOUT_S = 5.0
    _DEFAULT_POLL_INTERVAL_S = 0.05
    _DEFAULT_OPEN_WIDTH_TOLERANCE_MM = 2.0

    def __init__(
            self, ip: str, port: int = 502, hardware_enabled: bool = False,
            gripper: str = 'rg2', node=None):
        self.ip = ip
        self.port = port
        self.hardware_enabled = hardware_enabled
        self.gripper = gripper
        self._node = node
        self._client = None
        self._sim_width_mm = self.MAX_WIDTH_MM.get(gripper, 110.0)
        self._sim_grip_detected = False
        self.last_status = None
        self.last_width_mm = None
        self.last_grip_detected = None

    def _ensure_connected(self):
        if self._client is None:
            from pymodbus.client.sync import ModbusTcpClient
            self._client = ModbusTcpClient(
                self.ip, port=self.port, stopbits=1, bytesize=8, parity='E',
                baudrate=115200, timeout=1)
        try:
            if not self._client.connect():
                return None
        except Exception:
            return None
        return self._client

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

    def _read_busy_bit(self):
        try:
            client = self._ensure_connected()
            if client is None:
                return None
            response = client.read_holding_registers(address=268, count=1, unit=65)
            if self._response_failed(response):
                return None
            registers = getattr(response, 'registers', None)
            return None if not registers else bool(registers[0] & 0x01)
        except Exception:
            return None

    def _read_final_state(self):
        try:
            client = self._ensure_connected()
            if client is None:
                return None
            width_response = client.read_holding_registers(address=267, count=1, unit=65)
            status_response = client.read_holding_registers(address=268, count=1, unit=65)
            if self._response_failed(width_response) or self._response_failed(status_response):
                return None
            width_registers = getattr(width_response, 'registers', None)
            status_registers = getattr(status_response, 'registers', None)
            if not width_registers or not status_registers:
                return None
            width_mm = width_registers[0] / 10.0
            if not math.isfinite(width_mm):
                return None
            return width_mm, bool(status_registers[0] & 0x02)
        except Exception:
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
                address=0, values=[0, 0, self._CMD_STOP], unit=65)
        except Exception:
            return False
        if self._response_failed(response):
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
            response = client.write_registers(address=0, values=values, unit=65)
        except Exception:
            return RG2Status.COMMUNICATION_ERROR
        if self._response_failed(response):
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
        if width_mm < 0.0 or width_mm > max_width:
            return RG2Status.COMMUNICATION_ERROR
        self.last_width_mm = width_mm
        self.last_grip_detected = grip_detected
        return RG2Status.SUCCESS

    def open(self, goal_handle=None) -> bool:
        self.last_status = None
        self.last_width_mm = None
        self.last_grip_detected = None
        if self.gripper not in self.MAX_WIDTH_MM or self.gripper not in self.MAX_FORCE_N:
            self.last_status = RG2Status.INVALID_INPUT
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

        self.last_status = self._run_command(
            goal_handle,
            [int(max_force * 10), int(max_width * 10), self._CMD_GRIP_W_OFFSET],
            max_width)
        if self.last_status != RG2Status.SUCCESS:
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
            return False
        if not self.hardware_enabled:
            self._sim_width_mm = width_mm
            self._sim_grip_detected = True
            self.last_status = RG2Status.SUCCESS
            self.last_width_mm = width_mm
            self.last_grip_detected = True
            return True

        self.last_status = self._run_command(
            goal_handle,
            [int(force_n * 10), int(width_mm * 10), self._CMD_GRIP_W_OFFSET],
            self.MAX_WIDTH_MM[self.gripper])
        return self.last_status == RG2Status.SUCCESS

    def get_state(self):
        if not self.hardware_enabled:
            return self._sim_width_mm, self._sim_grip_detected
        return self._read_final_state() or (0.0, False)


__all__ = ['RG2Client', 'RG2Status']
