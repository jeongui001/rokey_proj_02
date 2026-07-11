import pytest
from robot_control.tools.kalman_replay import read_track_csv, write_replay_csv


def test_read_track_csv_parses_rows_and_casts_types(tmp_path):
    csv_path = tmp_path / 'track.csv'
    csv_path.write_text(
        'stamp_s,recv_monotonic_s,x,y,z,depth_valid\n'
        '0.0,100.0,0.10,0.20,0.05,True\n'
        '0.02,100.02,0.11,0.20,0.05,False\n'
    )
    rows = read_track_csv(str(csv_path))
    assert len(rows) == 2
    assert rows[0].stamp_s == pytest.approx(0.0)
    assert rows[0].recv_monotonic_s == pytest.approx(100.0)
    assert rows[0].x == pytest.approx(0.10)
    assert rows[0].depth_valid is True
    assert rows[1].depth_valid is False


def test_write_replay_csv_round_trips_via_read_track_csv_header(tmp_path):
    out_path = tmp_path / 'out.csv'
    records = [
        {'stamp_s': 0.0, 'w': 0.5, 'innovation_xy_m': 0.001},
        {'stamp_s': 0.02, 'w': 0.6, 'innovation_xy_m': 0.002},
    ]
    write_replay_csv(str(out_path), records)
    lines = out_path.read_text().strip().splitlines()
    assert lines[0] == 'stamp_s,w,innovation_xy_m'
    assert len(lines) == 3


def test_write_replay_csv_rejects_empty_records(tmp_path):
    with pytest.raises(ValueError):
        write_replay_csv(str(tmp_path / 'out.csv'), [])
