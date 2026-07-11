import pytest

from robot_control.rg2_client import RG2Client, RG2Status


class _FakeResponse:
    """registers가 있는/없는, isError()가 참/거짓인 pymodbus 응답을 흉내낸다."""

    def __init__(self, registers=None, is_error=False):
        self.registers = registers
        self._is_error = is_error

    def isError(self):
        return self._is_error


class _FakeModbusClient:
    """RG2Client 테스트용 범용 fake.

    read_busy_values: read_holding_registers(268) 호출마다 순서대로 소비되는
    busy(0/1) 목록 - 첫 호출은 명령 전 검사(_check_before_command)의 busy 확인,
    이후 호출은 명령 후 busy=0 대기(post-check)에 대응한다. 목록이 소진되면
    마지막 값을 계속 반환한다.
    final_width_mm/grip_detected: read_holding_registers(267/268)의 최종 상태 값.
    connect_ok/write_is_error/read_raises/write_raises: 연결·쓰기·읽기 실패를
    시뮬레이션한다.
    """

    def __init__(self, connect_ok=True, read_busy_values=(0,), final_width_mm=30.0,
                 grip_detected=True, write_is_error=False, read_raises_after=None,
                 write_raises=False, missing_registers=False):
        self.connect_ok = connect_ok
        self.read_busy_values = list(read_busy_values)
        self.final_width_mm = final_width_mm
        self.grip_detected = grip_detected
        self.write_is_error = write_is_error
        self.read_raises_after = read_raises_after  # 이 번째 read 호출부터 예외 발생
        self.write_raises = write_raises
        self.missing_registers = missing_registers
        self.write_calls = []
        self._busy_read_count = 0
        self._read_268_count = 0

    def connect(self):
        return self.connect_ok

    def write_registers(self, address, values, slave):
        self.write_calls.append((address, values, slave))
        if self.write_raises:
            raise RuntimeError('modbus 통신 오류 (simulated write)')
        return _FakeResponse(is_error=self.write_is_error)

    def read_holding_registers(self, address, count, slave):
        if address == 268:
            self._read_268_count += 1
            if (self.read_raises_after is not None
                    and self._read_268_count >= self.read_raises_after):
                raise RuntimeError('modbus 통신 오류 (simulated read)')
            idx = min(self._busy_read_count, len(self.read_busy_values) - 1)
            busy = self.read_busy_values[idx]
            self._busy_read_count += 1
            grip_bit = 1 if self.grip_detected else 0
            value = (grip_bit << 1) | (1 if busy else 0)
            return _FakeResponse(registers=[value])
        if address == 267:
            # width(267)는 busy(268) 폴링과 무관하게 최종 상태 확인 시에만 읽힌다 -
            # missing_registers는 그 최종 읽기 실패만 시뮬레이션한다.
            if self.missing_registers:
                return _FakeResponse(registers=[])
            return _FakeResponse(registers=[int(self.final_width_mm * 10)])
        return _FakeResponse(registers=[0])


class _FakeModbusClientDeviceId(_FakeModbusClient):
    """pymodbus 3.12+ 스타일(slave= 대신 device_id=)을 흉내낸다."""

    def write_registers(self, address, values, device_id):
        return super().write_registers(address, values, device_id)

    def read_holding_registers(self, address, count, device_id):
        return super().read_holding_registers(address, count, device_id)


class _FakeModbusClientUnit(_FakeModbusClient):
    """pymodbus 2.x 스타일(slave= 대신 unit=)을 흉내낸다."""

    def write_registers(self, address, values, unit):
        return super().write_registers(address, values, unit)

    def read_holding_registers(self, address, count, unit):
        return super().read_holding_registers(address, count, unit)


class _FakeModbusClientKwargsOnly(_FakeModbusClient):
    """pymodbus 2.x의 실제 read_holding_registers/write_registers 시그니처를
    흉내낸다 - unit을 명명된 파라미터가 아니라 **kwargs로만 받는다(실제로는
    ModbusPDU.__init__이 kwargs.get('unit', ...)으로 읽는다). unit/slave/device_id
    중 어느 것도 시그니처에 이름으로 노출되지 않는다는 점에서 위 fake들과 다르다."""

    def write_registers(self, address, values, **kwargs):
        return super().write_registers(address, values, kwargs.get('unit'))

    def read_holding_registers(self, address, count, **kwargs):
        return super().read_holding_registers(address, count, kwargs.get('unit'))


def _install_fake(monkeypatch, fake):
    monkeypatch.setattr('pymodbus.client.ModbusTcpClient', lambda *a, **kw: fake)
    return fake


class _FakeGoalHandle:
    def __init__(self, cancel_after=None):
        self._calls = 0
        self.cancel_after = cancel_after

    @property
    def is_cancel_requested(self):
        self._calls += 1
        if self.cancel_after is None:
            return False
        return self._calls >= self.cancel_after


# ---- pymodbus 버전별 unit/slave/device_id 키워드 인자 호환성 ----
# (2026-07-10: 팀원마다 설치된 pymodbus 버전이 달라 unit=/slave=/device_id=
# 중 하나만 하드코딩하면 다른 버전에서는 깨진다 - 실제 설치된 클라이언트의
# 시그니처를 보고 골라 쓰는 _resolve_modbus_id_kwarg를 검증한다.)

def test_resolve_modbus_id_kwarg_prefers_device_id_then_slave_then_unit():
    from robot_control.rg2_client import _resolve_modbus_id_kwarg

    assert _resolve_modbus_id_kwarg(_FakeModbusClientDeviceId()) == 'device_id'
    assert _resolve_modbus_id_kwarg(_FakeModbusClient()) == 'slave'
    assert _resolve_modbus_id_kwarg(_FakeModbusClientUnit()) == 'unit'


def test_resolve_modbus_id_kwarg_falls_back_to_unit_for_kwargs_only_signature():
    """실제 pymodbus 2.x의 read_holding_registers(self, address, count, **kwargs)처럼
    unit/slave/device_id 중 어느 것도 명명된 파라미터로 노출하지 않는 시그니처에서는
    unit으로 폴백해야 한다 - device_id로 폴백하면 unit 인자가 조용히 무시되어(2.x의
    ModbusPDU.__init__이 kwargs.get('unit', ...)로만 읽으므로) 실제 슬레이브 ID
    대신 0으로 통신하게 되어 Modbus 오류가 난다."""
    from robot_control.rg2_client import _resolve_modbus_id_kwarg

    assert _resolve_modbus_id_kwarg(_FakeModbusClientKwargsOnly()) == 'unit'


def test_close_and_get_state_work_with_pymodbus_3_12_plus_device_id_api(monkeypatch):
    """pymodbus 3.12+에서 slave=가 device_id=로 완전히 교체된 상황을 흉내내
    RG2Client가 여전히 정상 동작하는지 확인한다."""
    _install_fake(monkeypatch, _FakeModbusClientDeviceId())

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is True
    assert client.last_status == RG2Status.SUCCESS
    width_mm, grip_detected = client.get_state()
    assert width_mm == 30.0
    assert grip_detected is True


def test_close_and_get_state_work_with_legacy_pymodbus_unit_api(monkeypatch):
    """pymodbus 2.x 스타일(unit=)에서도 정상 동작하는지 확인한다."""
    _install_fake(monkeypatch, _FakeModbusClientUnit())

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is True
    assert client.last_status == RG2Status.SUCCESS


# ---- dry-run ----

def test_dry_run_open_does_not_touch_network_and_reports_max_width():
    client = RG2Client(ip='192.168.1.1', hardware_enabled=False)

    client.open()
    width_mm, grip_detected = client.get_state()

    assert client._client is None  # Modbus 연결을 시도하지 않았다
    assert width_mm == RG2Client.MAX_WIDTH_MM['rg2']
    assert grip_detected is False


def test_dry_run_close_simulates_grip_detected():
    client = RG2Client(ip='192.168.1.1', hardware_enabled=False)

    client.close(width_mm=30.0, force_n=20.0)
    width_mm, grip_detected = client.get_state()

    assert client._client is None
    assert width_mm == 30.0
    assert grip_detected is True


# ---- 정상 open/close: Modbus 쓰기 값 + busy 완료 + 최종 상태 확인 ----

def test_hardware_enabled_open_writes_expected_modbus_registers(monkeypatch):
    fake = _install_fake(monkeypatch, _FakeModbusClient(
        read_busy_values=[0, 0], final_width_mm=RG2Client.MAX_WIDTH_MM['rg2']))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.open()

    assert result is True
    assert client.last_status == RG2Status.SUCCESS
    assert fake.write_calls == [(0, [
        int(RG2Client.MAX_FORCE_N['rg2'] * 10),
        int(RG2Client.MAX_WIDTH_MM['rg2'] * 10),
        RG2Client._CMD_GRIP_W_OFFSET], 65)]


def test_hardware_enabled_close_writes_requested_width_and_force(monkeypatch):
    fake = _install_fake(monkeypatch, _FakeModbusClient(
        read_busy_values=[0, 0], final_width_mm=30.0))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is True
    assert client.last_status == RG2Status.SUCCESS
    assert fake.write_calls == [(0, [200, 300, RG2Client._CMD_GRIP_W_OFFSET], 65)]


def test_close_waits_for_busy_to_clear_then_succeeds(monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    # idx0: 명령 전 검사(busy=0) -> write -> idx1,2: 명령 후 대기(busy=1,1) -> idx3: 0
    fake = _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[0, 1, 1, 0]))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is True
    assert client.last_status == RG2Status.SUCCESS
    assert len(fake.write_calls) == 1


def test_close_waits_for_existing_busy_before_sending_new_command(monkeypatch):
    # 명령 전 검사: 이미 진행 중이던 busy(1,1)가 해제(0)될 때까지 기다린 뒤에만 쓴다.
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    fake = _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[1, 1, 0, 0]))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is True
    assert len(fake.write_calls) == 1  # busy 해제를 기다린 뒤 정확히 한 번만 썼다


# ---- 입력값 검사 (NaN/Inf, 범위, 지원하지 않는 gripper) ----

@pytest.mark.parametrize('width_mm,force_n', [
    (float('nan'), 20.0),
    (float('inf'), 20.0),
    (30.0, float('nan')),
    (30.0, float('-inf')),
    (-1.0, 20.0),           # 음수 폭
    (150.0, 20.0),          # RG2 최대 폭(110mm) 초과
    (30.0, -1.0),           # 음수 힘
    (30.0, 999.0),          # RG2 최대 힘(40N) 초과
])
def test_close_rejects_invalid_inputs_without_modbus_write(monkeypatch, width_mm, force_n):
    fake = _install_fake(monkeypatch, _FakeModbusClient())

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=width_mm, force_n=force_n)

    assert result is False
    assert client.last_status == RG2Status.INVALID_INPUT
    assert fake.write_calls == []  # 잘못된 값은 clamp하지 않고 Modbus를 전혀 건드리지 않는다


def test_dry_run_close_rejects_invalid_input_without_simulating_grip():
    client = RG2Client(ip='192.168.1.1', hardware_enabled=False)

    result = client.close(width_mm=float('nan'), force_n=20.0)

    assert result is False  # dry-run에도 같은 입력 검사가 적용된다
    assert client.last_status == RG2Status.INVALID_INPUT
    assert client._sim_grip_detected is False  # 잘못된 입력은 시뮬레이션 상태도 바꾸지 않는다


def test_open_and_close_reject_unsupported_gripper_name():
    client = RG2Client(ip='192.168.1.1', hardware_enabled=True, gripper='not_a_real_gripper')

    assert client.open() is False
    assert client.last_status == RG2Status.INVALID_INPUT
    assert client.close(width_mm=30.0, force_n=20.0) is False
    assert client.last_status == RG2Status.INVALID_INPUT


# ---- 연결/쓰기 응답 확인 ----

def test_close_fails_when_connect_returns_false(monkeypatch):
    fake = _install_fake(monkeypatch, _FakeModbusClient(connect_ok=False))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.COMMUNICATION_ERROR
    assert fake.write_calls == []  # 연결 자체가 안 됐으므로 쓰기 시도조차 하지 않는다


def test_close_fails_when_write_response_is_error(monkeypatch):
    fake = _install_fake(monkeypatch, _FakeModbusClient(write_is_error=True))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.COMMUNICATION_ERROR
    assert len(fake.write_calls) == 1  # 쓰기는 시도했지만 응답이 오류였다


def test_close_fails_when_write_raises(monkeypatch):
    _install_fake(monkeypatch, _FakeModbusClient(write_raises=True))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.COMMUNICATION_ERROR


# ---- timeout / 통신 오류 (busy 폴링 중) ----

def test_close_times_out_when_busy_never_clears(monkeypatch):
    import time as time_module
    clock = {'t': 0.0}
    monkeypatch.setattr(time_module, 'monotonic', lambda: clock['t'])

    def fake_sleep(_s):
        clock['t'] += 0.5  # 실제 대기 없이 시계만 빠르게 전진시킨다

    monkeypatch.setattr(time_module, 'sleep', fake_sleep)
    # idx0(명령 전)=0 -> write -> 이후(명령 후) 계속 busy=1
    _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[0, 1]))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)  # node 없음 -> 기본 timeout 5.0s
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.TIMEOUT


def test_close_returns_false_on_communication_error_during_busy_poll(monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    # 첫 read(명령 전 검사)는 성공(busy=0), 이후(명령 후 대기)부터 예외 발생.
    _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[0], read_raises_after=2))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.COMMUNICATION_ERROR


# ---- 명령 전 취소/Fault: write_registers 호출 0회 ----

def test_close_stops_waiting_immediately_when_cancel_already_requested(monkeypatch):
    fake = _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[1]))  # 항상 busy

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0, goal_handle=_FakeGoalHandle(cancel_after=1))

    assert result is False
    assert client.last_status == RG2Status.CANCELED
    assert fake.write_calls == []  # 취소가 이미 요청된 경우 Modbus 쓰기를 전혀 하지 않는다


def test_close_stops_waiting_when_node_safety_state_not_normal(monkeypatch):
    fake = _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[1]))

    class _Param:
        def __init__(self, value):
            self.value = value

    class _FakeNode:
        safety_state = 'FAULT'

        def get_parameter(self, name):
            values = {
                'rg2.command_timeout_s': 5.0,
                'rg2.poll_interval_s': 0.01,
                'rg2.open_width_tolerance_mm': 2.0,
                'rg2.connect_timeout_s': 2.0,
                'rg2.communication_retry_count': 0,
                'rg2.communication_retry_backoff_s': 0.5,
            }
            return _Param(values[name])

        def get_logger(self):
            class _L:
                def warn(self, *a, **k):
                    pass

                def error(self, *a, **k):
                    pass
            return _L()

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True, node=_FakeNode())
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.FAULT
    assert fake.write_calls == []  # Fault 상태에서도 Modbus 쓰기를 전혀 하지 않는다


# ---- 실행 중 취소/Fault -> command=8(stop) ----

def test_close_sends_stop_command_when_canceled_during_busy_wait(monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    # 호출 순서: #1 _check_before_command 자체 취소 확인(False),
    # #2 명령 전 busy 대기 루프의 취소 확인(False, 이어서 busy=0 읽고 통과),
    # write(1회) 후 #3 명령 후 busy 대기 루프의 취소 확인(True -> 취소 감지, busy를
    # 다시 읽기 전에 즉시 CANCELED 반환) -> stop 전송 -> stop 후 busy=0 확인(성공).
    fake = _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[0]))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    goal_handle = _FakeGoalHandle(cancel_after=3)
    result = client.close(width_mm=30.0, force_n=20.0, goal_handle=goal_handle)

    assert result is False
    assert client.last_status == RG2Status.CANCELED
    # 첫 번째 쓰기는 원래 close 명령, 두 번째는 stop(command=8)이어야 한다.
    assert len(fake.write_calls) == 2
    assert fake.write_calls[1][0] == 0
    assert fake.write_calls[1][1][2] == RG2Client._CMD_STOP


def test_close_reports_fault_when_stop_fails_after_cancel(monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    class _StopFailingClient(_FakeModbusClient):
        def write_registers(self, address, values, slave):
            self.write_calls.append((address, values, slave))
            if values[2] == RG2Client._CMD_STOP:
                raise RuntimeError('stop 쓰기 실패 (simulated)')
            return _FakeResponse(is_error=False)

    _install_fake(monkeypatch, _StopFailingClient(read_busy_values=[0]))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    goal_handle = _FakeGoalHandle(cancel_after=3)
    result = client.close(width_mm=30.0, force_n=20.0, goal_handle=goal_handle)

    assert result is False
    # stop 자체가 실패하면 취소 성공으로 가장하지 않고 FAULT용 실패로 보고한다.
    assert client.last_status == RG2Status.COMMUNICATION_ERROR


# ---- open 완료 후 최종 폭 확인 ----

def test_open_records_final_width_on_success(monkeypatch):
    _install_fake(monkeypatch, _FakeModbusClient(
        read_busy_values=[0, 0], final_width_mm=RG2Client.MAX_WIDTH_MM['rg2'], grip_detected=False))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.open()

    assert result is True
    assert client.last_width_mm == RG2Client.MAX_WIDTH_MM['rg2']
    assert client.last_grip_detected is False


def test_open_succeeds_when_final_width_exceeds_max_within_tolerance(monkeypatch):
    """실측 폭이 스펙 최대값을 open_width_tolerance_mm 이내로 넘어도(캘리브레이션 오차)
    정상 성공으로 처리해야 한다 - 실기에서 그리퍼가 실제로는 다 열렸는데도
    COMMUNICATION_ERROR로 오판되어 재시도 끝에 FAULT로 이어지던 문제(2026-07-11) 재현."""
    over_width = RG2Client.MAX_WIDTH_MM['rg2'] + 1.5  # 기본 tolerance(2.0mm) 이내 초과
    _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[0, 0], final_width_mm=over_width))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.open()

    assert result is True
    assert client.last_width_mm == over_width


def test_open_fails_when_final_width_out_of_model_range(monkeypatch):
    _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[0, 0], final_width_mm=9999.0))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.open()

    assert result is False
    assert client.last_status == RG2Status.COMMUNICATION_ERROR


def test_close_fails_when_final_state_read_returns_no_registers(monkeypatch):
    _install_fake(monkeypatch, _FakeModbusClient(read_busy_values=[0, 0], missing_registers=True))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.COMMUNICATION_ERROR


# ---- COMMUNICATION_ERROR 자동 재시도 ----

def _make_fake_node(retry_count=0, backoff_s=0.0, safety_state='NORMAL'):
    class _Param:
        def __init__(self, value):
            self.value = value

    class _Logger:
        def __init__(self):
            self.warnings = []
            self.errors = []

        def warn(self, msg, *a, **k):
            self.warnings.append(msg)

        def error(self, msg, *a, **k):
            self.errors.append(msg)

    class _FakeNode:
        def __init__(self):
            self.safety_state = safety_state
            self.logger = _Logger()

        def get_parameter(self, name):
            values = {
                'rg2.command_timeout_s': 5.0,
                'rg2.poll_interval_s': 0.01,
                'rg2.open_width_tolerance_mm': 2.0,
                'rg2.connect_timeout_s': 2.0,
                'rg2.communication_retry_count': retry_count,
                'rg2.communication_retry_backoff_s': backoff_s,
            }
            return _Param(values[name])

        def get_logger(self):
            return self.logger

    return _FakeNode()


def test_close_retries_once_on_communication_error_and_succeeds(monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    class _FlakyClient(_FakeModbusClient):
        def __init__(self):
            super().__init__(read_busy_values=[0, 0])
            self._connect_calls = 0

        def connect(self):
            self._connect_calls += 1
            return self._connect_calls > 1  # 첫 연결만 실패, 이후 성공

    fake = _install_fake(monkeypatch, _FlakyClient())
    node = _make_fake_node(retry_count=2, backoff_s=0.0)

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True, node=node)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is True
    assert client.last_status == RG2Status.SUCCESS
    assert len(fake.write_calls) == 1  # 실패한 첫 시도는 쓰기까지 가지도 못했다
    assert len(node.logger.warnings) == 1  # 재시도 1번만 있었다


def test_close_gives_up_after_exhausting_retries(monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _install_fake(monkeypatch, _FakeModbusClient(connect_ok=False))
    node = _make_fake_node(retry_count=2, backoff_s=0.0)

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True, node=node)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.COMMUNICATION_ERROR
    assert len(node.logger.warnings) == 2  # retry_count=2만큼 재시도했다


def test_close_does_not_retry_when_canceled(monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)
    _install_fake(monkeypatch, _FakeModbusClient())
    node = _make_fake_node(retry_count=2, backoff_s=0.0)
    gh = _FakeGoalHandle(cancel_after=1)  # 첫 확인부터 취소된 상태

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True, node=node)
    result = client.close(width_mm=30.0, force_n=20.0, goal_handle=gh)

    assert result is False
    assert client.last_status == RG2Status.CANCELED
    assert node.logger.warnings == []  # CANCELED는 재시도 대상이 아니다


def test_close_without_retry_configured_fails_immediately(monkeypatch):
    # node 없이(기존 테스트들과 동일) 생성하면 기본 재시도 횟수가 0이라 기존
    # 동작(즉시 실패)이 그대로 유지된다.
    fake = _install_fake(monkeypatch, _FakeModbusClient(connect_ok=False))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    result = client.close(width_mm=30.0, force_n=20.0)

    assert result is False
    assert client.last_status == RG2Status.COMMUNICATION_ERROR
    assert fake.write_calls == []


# ---- get_state 안전화 ----

def test_get_state_returns_safe_default_when_read_raises(monkeypatch):
    class _RaisingClient(_FakeModbusClient):
        def read_holding_registers(self, address, count, slave):
            raise RuntimeError('modbus 통신 오류 (simulated)')

    _install_fake(monkeypatch, _RaisingClient())

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)

    width_mm, grip_detected = client.get_state()  # 예외를 던지지 않아야 한다

    assert width_mm == 0.0
    assert grip_detected is False


def test_get_state_returns_safe_default_when_registers_missing(monkeypatch):
    _install_fake(monkeypatch, _FakeModbusClient(missing_registers=True))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)

    width_mm, grip_detected = client.get_state()

    assert width_mm == 0.0
    assert grip_detected is False


def test_get_state_returns_safe_default_when_connect_fails(monkeypatch):
    _install_fake(monkeypatch, _FakeModbusClient(connect_ok=False))

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)

    width_mm, grip_detected = client.get_state()

    assert width_mm == 0.0
    assert grip_detected is False
