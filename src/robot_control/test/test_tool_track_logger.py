import pytest
from robot_control.tools.tool_track_logger import _parse_args, format_row


class FakeStamp:
    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


class FakeHeader:
    def __init__(self, sec, nanosec):
        self.stamp = FakeStamp(sec, nanosec)


class FakePosition:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakePose:
    def __init__(self, x, y, z):
        self.position = FakePosition(x, y, z)


class FakeToolTrack:
    def __init__(self, sec, nanosec, x, y, z, depth_valid):
        self.header = FakeHeader(sec, nanosec)
        self.pose = FakePose(x, y, z)
        self.depth_valid = depth_valid


def test_format_row_extracts_stamp_position_and_depth_valid():
    msg = FakeToolTrack(sec=10, nanosec=500_000_000, x=0.1, y=0.2, z=0.3, depth_valid=True)
    row = format_row(msg, recv_monotonic_s=123.456)
    assert row == {
        'stamp_s': pytest.approx(10.5),
        'recv_monotonic_s': 123.456,
        'x': 0.1,
        'y': 0.2,
        'z': 0.3,
        'depth_valid': True,
    }


def test_format_row_casts_depth_valid_to_bool():
    msg = FakeToolTrack(sec=0, nanosec=0, x=0.0, y=0.0, z=0.0, depth_valid=0)
    row = format_row(msg, recv_monotonic_s=0.0)
    assert row['depth_valid'] is False


def test_parse_args_requires_out():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_parse_args_defaults_topic():
    args = _parse_args(['--out', '/tmp/x.csv'])
    assert args.topic == '/vision/tool_track'
    assert args.out == '/tmp/x.csv'
