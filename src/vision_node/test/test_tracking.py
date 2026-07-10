import math
import pytest
from vision_node.tracking import (
    pixel_to_camera_xyz, quaternion_to_rotation_matrix, transform_to_matrix,
    camera_to_base, is_approaching, ToolTracker, detection_center,
)


class FakeDetection:
    def __init__(self, class_name, score, x1, y1, x2, y2):
        self.class_name = class_name
        self.score = score
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


def test_detection_center_uses_keypoint_midpoint_when_confident():
    """pose 모델 검출: keypoint 2개가 유효하면 중점(=파지점)이 추적 기준이 된다."""
    d = FakeDetection('spanner', 0.9, 100, 100, 200, 140)
    d.kpt0_x, d.kpt0_y, d.kpt0_conf = 110.0, 130.0, 0.9
    d.kpt1_x, d.kpt1_y, d.kpt1_conf = 190.0, 110.0, 0.8
    assert detection_center(d) == pytest.approx((150.0, 120.0))


def test_detection_center_falls_back_to_bbox_when_kpt_low_conf():
    """keypoint 저신뢰(가림 등)면 bbox 중심으로 폴백한다."""
    d = FakeDetection('spanner', 0.9, 100, 100, 200, 140)
    d.kpt0_x, d.kpt0_y, d.kpt0_conf = 110.0, 130.0, 0.2
    d.kpt1_x, d.kpt1_y, d.kpt1_conf = 190.0, 110.0, 0.9
    assert detection_center(d) == pytest.approx((150.0, 120.0))


def test_detection_center_falls_back_when_no_kpt_fields():
    """kpt 필드가 아예 없는 구 box 모델 검출(테스트 스텁 포함)도 bbox 중심으로 동작한다."""
    d = FakeDetection('spanner', 0.9, 100, 100, 200, 140)
    assert detection_center(d) == pytest.approx((150.0, 120.0))


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
    assert depth_valid is True
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


def test_tracker_smooths_z_via_ema():
    """z(depth)는 EMA로 스무딩돼야 한다 - RealSense 스테레오 depth의 프레임간
    temporal jitter가 patch median(공간적 이상치 제거)만으로는 안 걸러지는 게
    체감 노이즈의 핵심 원인이었다(수정 전엔 z가 raw로 그대로 흘러나갔음)."""
    tracker = ToolTracker(alpha=0.5, beta=1.0)  # alpha_z 기본값=alpha
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy, bw, bh: (0.0, 0.0, 0.40, True), stamp=0.0)
    position, _, _, _ = tracker.update(
        [FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
        lambda cx, cy, bw, bh: (0.0, 0.0, 0.50, True), stamp=0.1)
    # raw z(0.50)를 그대로 쓰면 안 되고, alpha=0.5로 0.40과 0.50 사이(0.45)여야 한다
    assert position[2] == pytest.approx(0.45, abs=1e-6)


def test_tracker_does_not_smooth_xy():
    """x,y는 EMA를 적용하지 않고 raw 관측값을 그대로 써야 한다(2026-07-10, 사용자 지시:
    "ema가 z에는 적용되어야 하는데 xy에는 적용되면 안 돼") - 검출 좌표 자체는 이미
    안정적이라 스무딩하면 접근 중 지연만 유발한다. alpha를 낮게 줘도(스무딩이 켜져
    있었다면 값이 이전 값 쪽으로 끌렸을 것) x,y는 새 관측값으로 즉시 스냅해야 한다."""
    tracker = ToolTracker(alpha=0.1, beta=1.0, alpha_z=1.0)
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy, bw, bh: (0.0, 0.0, 0.40, True), stamp=0.0)
    position, _, _, _ = tracker.update(
        [FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
        lambda cx, cy, bw, bh: (1.0, 2.0, 0.50, True), stamp=0.1)
    assert position[0] == pytest.approx(1.0, abs=1e-6)
    assert position[1] == pytest.approx(2.0, abs=1e-6)


def test_tracker_alpha_z_override_can_smooth_depth_harder_than_xy():
    """depth 노이즈가 xy보다 크면 alpha_z를 따로 낮춰 더 세게 눌러야 한다."""
    tracker = ToolTracker(alpha=1.0, beta=1.0, alpha_z=0.1)
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy, bw, bh: (0.0, 0.0, 0.40, True), stamp=0.0)
    position, _, _, _ = tracker.update(
        [FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
        lambda cx, cy, bw, bh: (0.0, 0.0, 0.50, True), stamp=0.1)
    # alpha(xy)=1.0이라 x,y는 즉시 새 값으로 스냅하지만 z는 alpha_z=0.1로 거의 안 움직여야 함
    assert position[2] == pytest.approx(0.41, abs=1e-6)


def test_tracker_holds_last_valid_z_when_depth_invalid():
    tracker = ToolTracker()
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy, bw, bh: (0.0, 0.0, 0.05, True), stamp=0.0)
    position, _, depth_valid, _ = tracker.update(
        [FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
        lambda cx, cy, bw, bh: (0.1, 0.0, 999.0, False), stamp=0.1)
    assert depth_valid is False
    assert position[2] == pytest.approx(0.05, abs=1e-6)


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


def test_tracker_reset_clears_learned_offset():
    """set_mode 재진입(reset) 후에는 이전 도구의 오프셋이 새 도구에 새어들면 안 된다."""
    tracker = ToolTracker(alpha=1.0, beta=1.0)
    tracker.update([_kpt_det()], 'spanner', _px_reconstruct, stamp=0.0)
    assert tracker.kpt_offset is not None
    tracker.reset()
    assert tracker.kpt_offset is None
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
