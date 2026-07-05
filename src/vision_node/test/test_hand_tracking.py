import numpy as np
from vision_node.hand_tracking import detect_hand_wrist_pixel


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class FakeResult:
    def __init__(self, multi_hand_landmarks):
        self.multi_hand_landmarks = multi_hand_landmarks


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
