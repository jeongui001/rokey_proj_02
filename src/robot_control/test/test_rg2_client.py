from robot_control.rg2_client import RG2Client


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


def test_hardware_enabled_open_writes_expected_modbus_registers(monkeypatch):
    written = []

    class _FakeModbusClient:
        def __init__(self, *a, **kw):
            pass

        def connect(self):
            return True

        def write_registers(self, address, values, unit):
            written.append((address, values, unit))

    monkeypatch.setattr(
        'pymodbus.client.sync.ModbusTcpClient', _FakeModbusClient)

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    client.open()

    assert written == [(0, [
        int(RG2Client.MAX_FORCE_N['rg2'] * 10),
        int(RG2Client.MAX_WIDTH_MM['rg2'] * 10),
        RG2Client._CMD_GRIP_W_OFFSET], 65)]


def test_hardware_enabled_close_writes_requested_width_and_force(monkeypatch):
    written = []

    class _FakeModbusClient:
        def __init__(self, *a, **kw):
            pass

        def connect(self):
            return True

        def write_registers(self, address, values, unit):
            written.append((address, values, unit))

    monkeypatch.setattr(
        'pymodbus.client.sync.ModbusTcpClient', _FakeModbusClient)

    client = RG2Client(ip='192.168.1.1', hardware_enabled=True)
    client.close(width_mm=30.0, force_n=20.0)

    assert written == [(0, [200, 300, RG2Client._CMD_GRIP_W_OFFSET], 65)]
