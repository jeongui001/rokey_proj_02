from robot_control.tools.probe_speedl_stream import PhaseSegment, build_phase_plan


def test_phase1_constant_velocity_segment():
    segments = build_phase_plan(
        phase_duration_s=3.0, osc_period_s=1.0, osc_duration_s=0.0,
        pause_durations_s=[])
    assert segments[0] == PhaseSegment('publish', 'phase1_constant', 3.0, 1)


def test_oscillation_alternates_sign_each_period():
    segments = build_phase_plan(
        phase_duration_s=0.0, osc_period_s=1.0, osc_duration_s=4.0,
        pause_durations_s=[])
    osc_segments = [s for s in segments if s.label.startswith('phase2_osc_')]
    assert [s.sign for s in osc_segments] == [1, -1, 1, -1]
    assert all(s.duration_s == 1.0 for s in osc_segments)


def test_oscillation_last_segment_truncated_to_remaining_duration():
    segments = build_phase_plan(
        phase_duration_s=0.0, osc_period_s=1.0, osc_duration_s=2.5,
        pause_durations_s=[])
    osc_segments = [s for s in segments if s.label.startswith('phase2_osc_')]
    assert [s.duration_s for s in osc_segments] == [1.0, 1.0, 0.5]
    assert [s.sign for s in osc_segments] == [1, -1, 1]


def test_pause_resume_segments_preserve_order_and_alternate_kind():
    segments = build_phase_plan(
        phase_duration_s=0.0, osc_period_s=1.0, osc_duration_s=0.0,
        pause_durations_s=[0.5, 1.0, 2.0], pause_burst_s=1.0)
    phase3 = [s for s in segments if s.label.startswith('phase3_')]
    assert [s.kind for s in phase3] == [
        'publish', 'pause', 'publish', 'pause', 'publish', 'pause']
    assert [s.duration_s for s in phase3 if s.kind == 'pause'] == [0.5, 1.0, 2.0]
    assert all(s.duration_s == 1.0 for s in phase3 if s.kind == 'publish')


def test_full_plan_concatenates_all_three_phases_in_order():
    segments = build_phase_plan(
        phase_duration_s=3.0, osc_period_s=1.0, osc_duration_s=2.0,
        pause_durations_s=[0.5, 1.0])
    labels = [s.label for s in segments]
    assert labels == [
        'phase1_constant',
        'phase2_osc_0', 'phase2_osc_1',
        'phase3_burst_0', 'phase3_pause_0',
        'phase3_burst_1', 'phase3_pause_1',
    ]
