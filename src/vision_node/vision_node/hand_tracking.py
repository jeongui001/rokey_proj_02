"""MediaPipe 손목 픽셀 검출. 얇은 래퍼 — 모델 로딩과 랜드마크 인덱스만 감싼다."""

import cv2


def create_hands_detector():
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


def detect_hand_wrist_pixel(hands_detector, bgr_image):
    """bgr_image(np.ndarray)에서 손목(WRIST, landmark index 0) 픽셀 좌표 (px, py)를 반환.
    검출 실패 시 None."""
    h, w = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_image)
    if not result.multi_hand_landmarks:
        return None
    wrist = result.multi_hand_landmarks[0].landmark[0]
    return int(wrist.x * w), int(wrist.y * h)
