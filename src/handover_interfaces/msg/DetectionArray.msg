std_msgs/Header header
Detection2D[] detections
# 프로파일링(전 구간 타이밍은 vision_node가 VisionTiming으로 통합 발행)
float32 infer_ms           # model.predict 소요(NMS 포함)
float32 detect_latency_ms  # capture stamp -> 이 메시지 publish 직전
