# 런타임 파이프라인 구간 타이밍 - vision_node가 TRACK_TOOL 중 프레임당 1개 발행(/perception/timing).
# 용도: "처리량(FPS) 저하 vs 지연(latency) 누적" 판별. 구간 합(callback_ms까지)과 e2e_ms가
# 크게 벌어지면 큐 적체(오래된 프레임 처리 중)를 의심한다. tools/analyze_timing.py가
# rolling mean/std/p95 표를 만든다.
std_msgs/Header header        # 이 프레임의 capture stamp(RealSense 드라이버가 찍음) - 프레임 대조 키
float32 infer_ms              # tool_detection_node: model.predict 소요(NMS 포함)
float32 detect_latency_ms     # capture stamp -> DetectionArray publish 직전 (검출 노드 구간)
float32 sync_wait_ms          # DetectionArray publish -> 동기화 콜백 진입 (토픽 홉 + 4토픽 정렬 대기)
float32 tf_ms                 # 이미지 시각 TF lookup
float32 depth_ms              # depth 이미지 decode(imgmsg->m 단위 배열)
float32 track_ms              # tracker.update (후보별 3D 복원 포함)
float32 yaw_ms                # 그립 yaw 계산 (keypoint 벡터 또는 depth-PCA)
float32 debug_ms              # 디버그 이미지 encode + publish
float32 callback_ms           # _on_synced_images 진입->퇴장 (위 구간 합 + 파이썬 오버헤드)
float32 e2e_ms                # capture stamp -> 콜백 퇴장
uint8 published               # 이 프레임에 ToolTrack이 발행됐는가 (miss rate 계산용)
uint8 n_detections
