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


def test_tracker_holds_last_valid_z_when_depth_invalid():
    tracker = ToolTracker()
    tracker.update([FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
                    lambda cx, cy, bw, bh: (0.0, 0.0, 0.05, True), stamp=0.0)
    position, _, depth_valid, _ = tracker.update(
        [FakeDetection('spanner', 0.9, 0, 0, 0, 0)], 'spanner',
        lambda cx, cy, bw, bh: (0.1, 0.0, 999.0, False), stamp=0.1)
    assert depth_valid is False
    assert position[2] == pytest.approx(0.05, abs=1e-6)


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
