import numpy as np
from vision_node.hand_tracking import detect_hand, is_fist


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class FakeResult:
    def __init__(self, multi_hand_landmarks, multi_handedness=None):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness


class FakeHandsDetector:
    def __init__(self, result):
        self._result = result

    def process(self, rgb_image):
        return self._result


def _open_hand_landmarks():
    """편 손: 각 손가락 tip이 PIP보다 손목(0)에서 더 멀다(y가 클수록 멀다고 설정)."""
    lms = [FakeLandmark(0.5, 0.0) for _ in range(21)]
    lms[0] = FakeLandmark(0.5, 0.0)  # 손목
    # 검지/중지/약지/소지의 PIP(6/10/14/18)와 tip(8/12/16/20) - tip이 더 아래(멀리)
    for pip_i, tip_i in ((6, 8), (10, 12), (14, 16), (18, 20)):
        lms[pip_i] = FakeLandmark(0.5, 0.5)
        lms[tip_i] = FakeLandmark(0.5, 0.9)
    for mcp_i in (5, 9, 13, 17):  # 손바닥 MCP
        lms[mcp_i] = FakeLandmark(0.5, 0.3)
    return lms


def _fist_landmarks():
    """주먹: tip이 PIP보다 손목에 더 가깝다(tip이 위로 말려 올라옴)."""
    lms = [FakeLandmark(0.5, 0.0) for _ in range(21)]
    lms[0] = FakeLandmark(0.5, 0.0)
    for pip_i, tip_i in ((6, 8), (10, 12), (14, 16), (18, 20)):
        lms[pip_i] = FakeLandmark(0.5, 0.5)
        lms[tip_i] = FakeLandmark(0.5, 0.2)  # 손목(0.0)에 더 가까움
    for mcp_i in (5, 9, 13, 17):
        lms[mcp_i] = FakeLandmark(0.5, 0.3)
    return lms


def _as_tuples(fake_landmarks):
    return [(lm.x, lm.y) for lm in fake_landmarks]


def test_detect_hand_returns_none_when_no_hand():
    detector = FakeHandsDetector(FakeResult(None))
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert detect_hand(detector, image) is None


def test_detect_hand_returns_palm_center_scaled_by_image_size():
    lms = _open_hand_landmarks()  # MCP 평균 = (0.5, 0.3)
    detector = FakeHandsDetector(FakeResult([FakeHandLandmarks(lms)]))
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200

    palm_px, landmarks, confidence = detect_hand(detector, image)

    assert palm_px == (100, 30)  # (0.5*200, 0.3*100)
    assert len(landmarks) == 21
    assert confidence == 1.0  # handedness 없음 -> 기본 1.0


def test_detect_hand_uses_handedness_score_as_confidence():
    lms = _open_hand_landmarks()

    class _Cls:
        score = 0.87

    class _Handedness:
        classification = [_Cls()]

    detector = FakeHandsDetector(
        FakeResult([FakeHandLandmarks(lms)], multi_handedness=[_Handedness()]))
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    _, _, confidence = detect_hand(detector, image)
    assert confidence == 0.87


def test_is_fist_true_for_clenched_hand():
    assert is_fist(_as_tuples(_fist_landmarks())) is True


def test_is_fist_false_for_open_hand():
    assert is_fist(_as_tuples(_open_hand_landmarks())) is False


def test_is_fist_false_for_insufficient_landmarks():
    assert is_fist([(0.0, 0.0)] * 5) is False
    assert is_fist(None) is False
