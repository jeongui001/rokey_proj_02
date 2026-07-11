import pytest
from robot_control.tools.kalman_replay import TrackRow, read_track_csv, write_replay_csv, replay_kalman


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


def _static_rows(n, x=0.5, y=0.1, z=0.05, dt=0.02):
    return [TrackRow(stamp_s=i * dt, recv_monotonic_s=100.0 + i * dt,
                      x=x, y=y, z=z, depth_valid=True) for i in range(n)]


def test_replay_kalman_first_row_has_no_innovation():
    rows = _static_rows(3)
    records = replay_kalman(rows, q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0)
    assert records[0]['innovation_xy_m'] is None
    assert records[0]['x'] == pytest.approx(0.5)


def test_replay_kalman_converges_toward_static_measurement():
    rows = _static_rows(50)
    records = replay_kalman(rows, q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0)
    last = records[-1]
    assert last['x'] == pytest.approx(0.5, abs=1e-3)
    assert last['y'] == pytest.approx(0.1, abs=1e-3)


def test_replay_kalman_holds_z_when_depth_invalid():
    rows = _static_rows(5, z=0.05)
    rows.append(TrackRow(stamp_s=0.10, recv_monotonic_s=100.10, x=0.5, y=0.1, z=999.0, depth_valid=False))
    records = replay_kalman(rows, q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0)
    assert records[-1]['z'] == pytest.approx(0.05, abs=1e-6)
