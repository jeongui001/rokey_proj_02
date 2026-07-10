"""MediaPipe 손 검출 + 주먹 판정. 얇은 래퍼 — 모델 로딩과 랜드마크 인덱스만 감싼다.

손 위치는 손목(0)이 아니라 손바닥 중심(MCP 관절 평균)을 쓴다 — 주먹을 쥐어도 크게
움직이지 않아 손목보다 추종 목표로 안정적이다. 주먹 판정(is_fist)은 랜드마크만 받는
순수 함수라 mediapipe 없이도 유닛테스트가 된다."""

import cv2

# MediaPipe Hands 랜드마크 인덱스(공식 스펙)
WRIST = 0
# 손바닥 중심 추정에 쓰는 MCP(손가락 첫마디) 관절들 - 검지/중지/약지/소지
PALM_MCP_IDS = (5, 9, 13, 17)
# 주먹 판정용 비-엄지 4손가락의 끝(tip)과 두번째 마디(PIP)
FINGER_TIP_IDS = (8, 12, 16, 20)
FINGER_PIP_IDS = (6, 10, 14, 18)
# 4손가락 중 이만큼 이상 굽으면 주먹으로 본다(엄지는 제외 - 굽힘 방향이 달라 불안정)
FIST_MIN_CURLED_FINGERS = 3


def create_hands_detector():
    # mediapipe는 모듈 최상단이 아니라 여기서 임포트한다 - 설치 안 된 환경에서도
    # 이 파일의 다른 부분(detect_hand/is_fist)은 테스트 가능하게 하기 위함.
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


def _distance(a, b):
    """정규화 좌표 (x, y) 두 점 사이 유클리드 거리."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def is_fist(landmarks):
    """정규화 랜드마크 21개 [(x, y), ...]로 주먹 여부를 판정한다.

    비-엄지 4손가락 각각에 대해 tip이 PIP보다 손목(WRIST)에 더 가까우면 굽힘으로 보고,
    FIST_MIN_CURLED_FINGERS개 이상 굽으면 True. 손목 기준 상대거리라 손 방향(회전)에
    비교적 강건하다. 랜드마크가 부족하면 False."""
    if landmarks is None or len(landmarks) < 21:
        return False
    wrist = landmarks[WRIST]
    curled = 0
    for tip_i, pip_i in zip(FINGER_TIP_IDS, FINGER_PIP_IDS):
        if _distance(landmarks[tip_i], wrist) < _distance(landmarks[pip_i], wrist):
            curled += 1
    return curled >= FIST_MIN_CURLED_FINGERS


def _hand_confidence(result):
    """검출 신뢰도를 0.0~1.0으로 반환. MediaPipe Hands는 랜드마크별 검출 점수를 직접
    주지 않으므로 handedness(좌/우 분류) 점수를 대용으로 쓰고, 없으면 1.0."""
    handedness = getattr(result, 'multi_handedness', None)
    if handedness:
        try:
            return float(handedness[0].classification[0].score)
        except (IndexError, AttributeError):
            pass
    return 1.0


def detect_hand(hands_detector, bgr_image):
    """bgr_image(np.ndarray)에서 손을 검출해 (palm_px, landmarks, confidence)를 반환.
    검출 실패 시 None.

    - palm_px: 손바닥 중심 픽셀 (px, py) — PALM_MCP_IDS 평균을 이미지 크기로 되돌린 값
    - landmarks: 정규화(0~1) 21개 [(x, y), ...] — is_fist 등에 그대로 넘길 수 있음
    - confidence: 검출 신뢰도(0.0~1.0)
    MediaPipe는 랜드마크를 0~1 정규화 값으로 주므로 실제 크기(h, w)를 곱해 픽셀로 되돌린다."""
    h, w = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # MediaPipe는 RGB 입력을 기대
    result = hands_detector.process(rgb_image)
    if not result.multi_hand_landmarks:
        return None
    lms = result.multi_hand_landmarks[0].landmark
    landmarks = [(lm.x, lm.y) for lm in lms]
    cx = sum(landmarks[i][0] for i in PALM_MCP_IDS) / len(PALM_MCP_IDS)
    cy = sum(landmarks[i][1] for i in PALM_MCP_IDS) / len(PALM_MCP_IDS)
    palm_px = (int(cx * w), int(cy * h))
    return palm_px, landmarks, _hand_confidence(result)
