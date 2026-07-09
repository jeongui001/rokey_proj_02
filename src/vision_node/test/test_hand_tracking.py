import numpy as np
import pytest
from vision_node.hand_tracking import detect_hand_landmarks, detect_hand_wrist_pixel, is_fist


class FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class FakeClassification:
    def __init__(self, score):
        self.score = score


class FakeHandedness:
    def __init__(self, score):
        self.classification = [FakeClassification(score)]


class FakeResult:
    def __init__(self, multi_hand_landmarks, multi_handedness=None):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness


class FakeHandsDetector:
    def __init__(self, result):
        self._result = result

    def process(self, rgb_image):
        return self._result


def test_returns_none_when_no_hand_detected():
    detector = FakeHandsDetector(FakeResult(None))
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert detect_hand_wrist_pixel(detector, image) is None


def test_returns_wrist_pixel_scaled_by_image_size():
    landmarks = [FakeLandmark(0.5, 0.25)] + [FakeLandmark(0.0, 0.0)] * 20
    detector = FakeHandsDetector(FakeResult([FakeHandLandmarks(landmarks)]))
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200

    px, py = detect_hand_wrist_pixel(detector, image)

    assert px == 100
    assert py == 25


def test_detect_hand_landmarks_returns_none_and_zero_confidence_when_missing():
    detector = FakeHandsDetector(FakeResult(None))
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    landmarks, confidence = detect_hand_landmarks(detector, image)

    assert landmarks is None
    assert confidence == 0.0


def test_detect_hand_landmarks_returns_handedness_score_as_confidence():
    landmarks_in = [FakeLandmark(0.0, 0.0)] * 21
    detector = FakeHandsDetector(FakeResult(
        [FakeHandLandmarks(landmarks_in)], multi_handedness=[FakeHandedness(0.87)]))
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    landmarks, confidence = detect_hand_landmarks(detector, image)

    assert landmarks is landmarks_in
    assert confidence == pytest.approx(0.87)


def _make_landmarks(finger_pip_tip_distances):
    """4개 손가락(index/middle/ring/pinky)의 (pip 거리, tip 거리)를 손목(0,0,0) 기준
    +x 방향에 배치한 21개짜리 랜드마크 리스트를 만든다. is_fist는 pip/tip 거리
    비교만 쓰므로 mcp/dip는 pip-tip 사이 값으로 채운다."""
    landmarks = [FakeLandmark(0.0, 0.0) for _ in range(21)]
    finger_joint_groups = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    for (pip_d, tip_d), (mcp_i, pip_i, dip_i, tip_i) in zip(
            finger_pip_tip_distances, finger_joint_groups):
        landmarks[mcp_i] = FakeLandmark(pip_d * 0.5, 0.0)
        landmarks[pip_i] = FakeLandmark(pip_d, 0.0)
        landmarks[dip_i] = FakeLandmark((pip_d + tip_d) / 2, 0.0)
        landmarks[tip_i] = FakeLandmark(tip_d, 0.0)
    return landmarks


def test_is_fist_true_when_all_four_fingers_curled():
    # 4개 손가락 모두 tip이 pip보다 손목에 더 가까움(접힘)
    landmarks = _make_landmarks([(0.10, 0.05)] * 4)
    assert is_fist(landmarks) is True


def test_is_fist_false_when_all_fingers_extended():
    # 4개 손가락 모두 tip이 pip보다 손목에서 더 멂(펴짐)
    landmarks = _make_landmarks([(0.10, 0.20)] * 4)
    assert is_fist(landmarks) is False


def test_is_fist_false_when_one_finger_still_extended():
    # 3개는 접혔지만 1개(pinky)는 펴진 상태 - 주먹이 아니다.
    landmarks = _make_landmarks([(0.10, 0.05), (0.10, 0.05), (0.10, 0.05), (0.10, 0.20)])
    assert is_fist(landmarks) is False
