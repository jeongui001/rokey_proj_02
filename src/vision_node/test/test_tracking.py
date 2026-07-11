import math
import pytest
from vision_node.tracking import (
    pixel_to_camera_xyz, quaternion_to_rotation_matrix, transform_to_matrix,
    camera_to_base, is_approaching, ToolTracker,
)


class FakeDetection:
    def __init__(self, class_name, score, x1, y1, x2, y2):
        self.class_name = class_name
        self.score = score
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


def test_pixel_to_camera_xyz_center_pixel_is_zero_xy():
    x, y, z = pixel_to_camera_xyz(320, 240, 0.5, fx=600, fy=600, ppx=320, ppy=240)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.5)


def test_quaternion_identity_is_identity_matrix():
    r = quaternion_to_rotation_matrix(0, 0, 0, 1)
    assert r[0][0] == pytest.approx(1.0)
    assert r[1][1] == pytest.approx(1.0)
    assert r[2][2] == pytest.approx(1.0)
    assert r[0][1] == pytest.approx(0.0)


def test_camera_to_base_applies_translation():
    tf_matrix = transform_to_matrix((1.0, 2.0, 3.0), (0, 0, 0, 1))
    x, y, z = camera_to_base((0.1, 0.2, 0.3), tf_matrix)
    assert (x, y, z) == pytest.approx((1.1, 2.2, 3.3))


def test_is_approaching_true_when_moving_toward_ref():
    assert is_approaching((0.5, 0.0), (0.1, 0.0), (1.0, 0.0)) is True


def test_is_approaching_false_when_moving_away():
    assert is_approaching((0.5, 0.0), (-0.1, 0.0), (1.0, 0.0)) is False


def test_tracker_returns_none_when_no_matching_class():
    tracker = ToolTracker()
    dets = [FakeDetection('hammer', 0.9, 0, 0, 10, 10)]
    result = tracker.update(
        dets, 'spanner', lambda cx, cy, bw, bh: (0, 0, 0.05, True), stamp=0.0)
    assert result is None


def test_tracker_first_frame_uses_highest_score_and_zero_velocity():
    tracker = ToolTracker()
    dets = [
        FakeDetection('spanner', 0.5, 0, 0, 10, 10),
        FakeDetection('spanner', 0.9, 20, 20, 30, 30),
    ]

    def reconstruct(cx, cy, bbox_w, bbox_h):
        return (cx / 100.0, cy / 100.0, 0.05, True)

    position, velocity, depth_valid, chosen_det = tracker.update(
        dets, 'spanner', reconstruct, stamp=0.0)
    assert position == pytest.approx((0.25, 0.25, 0.05))
    assert velocity == pytest.approx((0.0, 0.0))
    # keypoint가 없는 FakeDetection은 bbox 모드로 잡히고, bbox는 mid가 아니라서
    # depth_valid가 False로 강제된다(2026-07-11 - mid가 아닌 z는 신뢰하지 않음).
    assert depth_valid is False
    assert chosen_det is dets[1]  # 최고 score 검출의 원본이 그대로 돌아와야 한다


def test_tracker_second_frame_estimates_velocity():
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    dets1 = [FakeDetection('spanner', 0.9, 0, 0, 0, 0)]
    dets2 = [FakeDetection('spanner', 0.9, 0, 0, 0, 0)]

    tracker.update(dets1, 'spanner', lambda cx, cy, bw, bh: (0.0, 0.0, 0.05, True), stamp=0.0)
    position, velocity, _, _ = tracker.update(
        dets2, 'spanner', lambda cx, cy, bw, bh: (0.1, 0.0, 0.05, True), stamp=1.0)

    assert position[0] == pytest.approx(0.1, abs=1e-6)
    assert velocity[0] == pytest.approx(0.1, abs=1e-6)


def test_tracker_holds_last_valid_z_when_depth_invalid():
    tracker = ToolTracker()
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy, bw, bh: (0.0, 0.0, 0.05, True), stamp=0.0)
    position, _, depth_valid, _ = tracker.update(
        [FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
        lambda cx, cy, bw, bh: (0.1, 0.0, 999.0, False), stamp=0.1)
    assert depth_valid is False
    assert position[2] == pytest.approx(0.05, abs=1e-6)


def test_tracker_returns_none_when_reconstruct_fn_returns_none_for_all_candidates():
    # vision_node.py의 reconstruct()는 만들어낼 z가 없으면(첫 프레임부터 depth 무효) 후보를
    # None으로 버린다(2026-07-08 실기 사고 이후 수정) - 그 결과 여기서도 유효 후보가 하나도
    # 없으면 검출 자체가 없었던 것과 동일하게 None을 반환해야 한다.
    tracker = ToolTracker()
    dets = [FakeDetection('spanner', 0.9, 0, 0, 10, 10)]
    result = tracker.update(dets, 'spanner', lambda cx, cy, bw, bh: None, stamp=0.0)
    assert result is None


def _kpt_det(class_name='spanner', score=0.9, bbox=(100, 100, 200, 140),
             p0=(110.0, 130.0, 0.9), p1=(190.0, 110.0, 0.8)):
    d = FakeDetection(class_name, score, *bbox)
    d.kpt0_x, d.kpt0_y, d.kpt0_conf = p0
    d.kpt1_x, d.kpt1_y, d.kpt1_conf = p1
    return d


def _px_reconstruct(cx, cy, bw, bh):
    """픽셀을 1/1000 스케일로 3D에 대응시키는 스텁 - 좌표 검증이 쉬워진다."""
    return (cx / 1000.0, cy / 1000.0, 0.5, True)


def test_tracker_single_kpt_uses_learned_offset():
    """근접 시 p1이 화면 밖으로 잘려 conf가 0으로 떨어져도(라이브런 실측 시나리오),
    두 kpt가 보이던 프레임에서 학습한 오프셋으로 파지점을 유지해야 한다 -
    잘린 bbox 중심으로 폴백하면 위치가 위로 편향된다(수정 전 버그)."""
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    # 1프레임: 두 kpt 유효 - mid=(150,120) -> (0.15,0.12), p0->mid 오프셋 학습
    tracker.update([_kpt_det()], 'spanner', _px_reconstruct, stamp=0.0)
    # 2프레임: p1 잘림(conf~0), bbox도 하단으로 부풀어 중심이 어긋난 상태
    d = _kpt_det(bbox=(100, 100, 200, 240), p1=(190.0, 110.0, 0.002))
    position, _, _, _ = tracker.update([d], 'spanner', _px_reconstruct, stamp=0.1)
    # p0(110,130) -> (0.11,0.13) + 오프셋(0.04,-0.01) = 원래 파지점 (0.15,0.12)
    assert position[0] == pytest.approx(0.15, abs=1e-6)
    assert position[1] == pytest.approx(0.12, abs=1e-6)


def test_tracker_single_kpt_p1_only_uses_negated_offset():
    """반대로 p0(머리)가 잘린 경우 - 파지점은 중점이므로 p1 기준 오프셋은 부호 반전."""
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    tracker.update([_kpt_det()], 'spanner', _px_reconstruct, stamp=0.0)
    d = _kpt_det(p0=(110.0, 130.0, 0.003))
    position, _, _, _ = tracker.update([d], 'spanner', _px_reconstruct, stamp=0.1)
    # p1(190,110) -> (0.19,0.11) - 오프셋(0.04,-0.01) = (0.15,0.12)
    assert position[0] == pytest.approx(0.15, abs=1e-6)
    assert position[1] == pytest.approx(0.12, abs=1e-6)


def test_tracker_single_kpt_without_history_falls_back_to_bbox():
    """오프셋 이력이 없으면(두 kpt로 본 적 없음) 저신뢰 kpt xy는 오염 가능성이 있어
    쓰지 않고 기존 bbox 중심 폴백을 유지한다."""
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    d = _kpt_det(bbox=(100, 100, 300, 140), p1=(190.0, 110.0, 0.002))
    position, _, _, _ = tracker.update([d], 'spanner', _px_reconstruct, stamp=0.0)
    # bbox 중심 (200,120) -> (0.2,0.12). p0 anchor(0.11)가 아니어야 한다
    assert position[0] == pytest.approx(0.2, abs=1e-6)


def test_tracker_p0_mode_freezes_z_at_last_mid_value():
    """mid가 아닌 모드(p0/p1/bbox)의 raw z는 신뢰하지 않고 마지막 mid z에 고정해야
    한다(2026-07-11) - p0/p1/bbox의 raw z는 파지점이 아닌 kpt/bbox 중심 depth라
    노이즈가 커서(실기: 이동 물체 픽에서 p0 구간 5초 동안 raw z가 1.6~12.3mm로
    흔들리며 그 위로도 벗어난 값에 정착 - 허공을 잡음) 스무딩만으로는 못 거른다.
    컨베이어 평면 가정상 z는 접근 중 안 변해야 하므로 mid에서 확정한 값을 계속
    믿는 게 재측정보다 낫다."""
    def reconstruct(cx, cy, bw, bh):
        z = 0.55 if (cx, cy) == (110.0, 130.0) and calls['n'] > 0 else 0.40
        return (cx / 1000.0, cy / 1000.0, z, True)

    calls = {'n': 0}
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    tracker.update([_kpt_det()], 'spanner', reconstruct, stamp=0.0)  # mid, z=0.40
    assert tracker.position[2] == pytest.approx(0.40, abs=1e-6)

    calls['n'] = 1
    d = _kpt_det(bbox=(100, 100, 200, 240), p1=(190.0, 110.0, 0.002))
    position, _, depth_valid, _ = tracker.update([d], 'spanner', reconstruct, stamp=0.1)
    assert tracker.last_mode == 'p0'
    # raw z=0.55이지만 mid가 아니므로 무시되고 마지막 mid z(0.40)에 고정돼야 한다
    assert position[2] == pytest.approx(0.40, abs=1e-6)
    assert depth_valid is False


def test_tracker_uses_raw_z_when_no_mid_ever_seen():
    """mid를 한 번도 못 거친 채로 시작부터 p0/p1/bbox면(얼릴 last_mid_z가 없음)
    어쩔 수 없이 raw z를 alpha_z_offset_mode로 스무딩해 쓴다."""
    tracker = ToolTracker(alpha=1.0, beta=1.0, alpha_z_offset_mode=0.1)
    d = _kpt_det(bbox=(100, 100, 300, 140), p1=(190.0, 110.0, 0.002))
    position, _, depth_valid, _ = tracker.update([d], 'spanner', _px_reconstruct, stamp=0.0)
    assert tracker.last_mode == 'bbox'
    assert position[2] == pytest.approx(0.5, abs=1e-6)
    assert depth_valid is False


def test_tracker_last_mode_is_mid_when_both_kpts_confident():
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    tracker.update([_kpt_det()], 'spanner', _px_reconstruct, stamp=0.0)
    assert tracker.last_mode == 'mid'


def test_tracker_last_mode_falls_back_to_bbox_without_offset_history():
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    d = _kpt_det(bbox=(100, 100, 300, 140), p1=(190.0, 110.0, 0.002))
    tracker.update([d], 'spanner', _px_reconstruct, stamp=0.0)
    assert tracker.last_mode == 'bbox'


def test_tracker_last_mode_is_p0_when_offset_learned_and_p1_lost():
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    tracker.update([_kpt_det()], 'spanner', _px_reconstruct, stamp=0.0)
    d = _kpt_det(bbox=(100, 100, 200, 240), p1=(190.0, 110.0, 0.002))
    tracker.update([d], 'spanner', _px_reconstruct, stamp=0.1)
    assert tracker.last_mode == 'p0'


def test_tracker_reset_clears_learned_offset():
    """set_mode 재진입(reset) 후에는 이전 도구의 오프셋이 새 도구에 새어들면 안 된다."""
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    tracker.update([_kpt_det()], 'spanner', _px_reconstruct, stamp=0.0)
    assert tracker.kpt_offset is not None
    assert tracker.last_mid_z is not None
    tracker.reset()
    assert tracker.kpt_offset is None
    assert tracker.last_mode is None
    assert tracker.last_mid_z is None
    d = _kpt_det(bbox=(100, 100, 300, 140), p1=(190.0, 110.0, 0.002))
    position, _, _, _ = tracker.update([d], 'spanner', _px_reconstruct, stamp=0.1)
    assert position[0] == pytest.approx(0.2, abs=1e-6)  # bbox 폴백


def test_tracker_picks_nearest_candidate_to_previous_position():
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy, bw, bh: (0.0, 0.0, 0.05, True), stamp=0.0)

    dets = [
        FakeDetection('spanner', 0.9, 100, 0, 100, 0),
        FakeDetection('spanner', 0.9, 1, 0, 1, 0),
    ]

    def reconstruct(cx, cy, bbox_w, bbox_h):
        return (1.0, 0.0, 0.05, True) if cx == 100 else (0.01, 0.0, 0.05, True)

    position, _, _, chosen_det = tracker.update(dets, 'spanner', reconstruct, stamp=0.1)
    assert position[0] == pytest.approx(0.01, abs=1e-3)
    assert chosen_det is dets[1]  # 최근접 후보의 원본 검출이 돌아와야 한다
