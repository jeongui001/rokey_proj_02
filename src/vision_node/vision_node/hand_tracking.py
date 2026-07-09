"""MediaPipe 손 랜드마크 검출 + 주먹 판별. 얇은 래퍼 - 모델 로딩과 랜드마크 인덱스만 감싼다."""

import cv2

WRIST = 0
# (mcp, pip, dip, tip) - 엄지 제외 4개 손가락(MediaPipe Hands 랜드마크 인덱스 스펙)
_FINGER_JOINTS = (
    (5, 6, 7, 8),      # index
    (9, 10, 11, 12),   # middle
    (13, 14, 15, 16),  # ring
    (17, 18, 19, 20),  # pinky
)


def create_hands_detector():
    # mediapipe는 모듈 최상단이 아니라 여기서 임포트한다 - 설치 안 된 환경에서도
    # 이 파일의 다른 부분(is_fist 등)은 테스트 가능하게 하기 위함.
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


def detect_hand_landmarks(hands_detector, bgr_image):
    """bgr_image(np.ndarray)에서 한 손의 21개 랜드마크(정규화 x,y,z)와 신뢰도를 반환한다.
    검출 실패 시 (None, 0.0). MediaPipe 좌표는 0~1 정규화된 값이다.

    신뢰도는 MediaPipe Hands가 랜드마크별로는 주지 않아 손 자체의 좌/우 분류
    신뢰도(multi_handedness)를 대신 쓴다 - 검출 품질의 대략적인 지표로 충분하다."""
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # MediaPipe는 RGB 입력을 기대
    result = hands_detector.process(rgb_image)
    if not result.multi_hand_landmarks:
        return None, 0.0
    landmarks = result.multi_hand_landmarks[0].landmark
    confidence = 0.5
    if result.multi_handedness:
        confidence = float(result.multi_handedness[0].classification[0].score)
    return landmarks, confidence


def detect_hand_wrist_pixel(hands_detector, bgr_image):
    """bgr_image(np.ndarray)에서 손목(WRIST, landmark index 0) 픽셀 좌표 (px, py)를 반환.
    검출 실패 시 None. MediaPipe 랜드마크는 0~1 정규화 값이므로 실제 이미지 크기(h, w)를
    곱해 픽셀 좌표로 되돌린다."""
    h, w = bgr_image.shape[:2]
    landmarks, _ = detect_hand_landmarks(hands_detector, bgr_image)
    if landmarks is None:
        return None
    wrist = landmarks[WRIST]
    return int(wrist.x * w), int(wrist.y * h)


def _distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def is_fist(landmarks) -> bool:
    """MediaPipe Hands 21개 랜드마크로 주먹 여부를 판별한다(엄지 제외 4개 손가락).

    손목 기준 거리 비교(이미지 평면 회전에 무관) - 손가락이 접히면 끝(TIP)이 중간
    관절(PIP)보다 손목에 더 가까워진다. 4개 손가락 모두 접혀야 주먹으로 판정한다.
    엄지는 다른 관절 구조 때문에 이 비교가 잘 맞지 않아 제외했다(실기 확인 후
    필요하면 별도 기준으로 추가한다)."""
    wrist = landmarks[WRIST]
    for _mcp, pip_idx, _dip, tip_idx in _FINGER_JOINTS:
        tip_dist = _distance(landmarks[tip_idx], wrist)
        pip_dist = _distance(landmarks[pip_idx], wrist)
        if tip_dist >= pip_dist:
            return False
    return True
