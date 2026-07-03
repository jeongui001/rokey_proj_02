import pytest
from robot_control.rg2_client import RG2Client


def test_stub_methods_raise_not_implemented():
    client = RG2Client(ip='192.168.1.1')
    with pytest.raises(NotImplementedError):
        client.open()
    with pytest.raises(NotImplementedError):
        client.close(30.0, 20.0)
    with pytest.raises(NotImplementedError):
        client.get_state()
