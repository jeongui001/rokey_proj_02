"""MediaPipe 손목 픽셀 검출. 얇은 래퍼 — 모델 로딩과 랜드마크 인덱스만 감싼다."""

import cv2


def create_hands_detector():
    # mediapipe는 모듈 최상단이 아니라 여기서 임포트한다 - 설치 안 된 환경에서도
    # 이 파일의 다른 부분(detect_hand_wrist_pixel)은 테스트 가능하게 하기 위함.
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


def detect_hand_wrist_pixel(hands_detector, bgr_image):
    """bgr_image(np.ndarray)에서 손목(WRIST, landmark index 0) 픽셀 좌표 (px, py)를 반환.
    검출 실패 시 None. MediaPipe는 랜드마크 좌표를 0~1 정규화된 값으로 주므로
    실제 이미지 크기(h, w)를 곱해 픽셀 좌표로 되돌린다."""
    h, w = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # MediaPipe는 RGB 입력을 기대
    result = hands_detector.process(rgb_image)
    if not result.multi_hand_landmarks:
        return None
    wrist = result.multi_hand_landmarks[0].landmark[0]  # 0번 = 손목(MediaPipe Hands 스펙)
    return int(wrist.x * w), int(wrist.y * h)
