import pytest

from stt_node.wakeup_word import _rms_to_level


def test_rms_to_level_zero_is_silence():
    assert _rms_to_level(0.0) == 0.0


def test_rms_to_level_full_scale_is_one():
    assert _rms_to_level(32768.0) == pytest.approx(1.0)


def test_rms_to_level_clips_above_full_scale():
    assert _rms_to_level(100000.0) == pytest.approx(1.0)


def test_rms_to_level_is_monotonic():
    low = _rms_to_level(50.0)
    mid = _rms_to_level(500.0)
    high = _rms_to_level(5000.0)
    assert 0.0 <= low < mid < high <= 1.0
