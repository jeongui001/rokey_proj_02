std_msgs/Header header       # base_link 기준, 관측 시각(시간 정합)
geometry_msgs/Pose pose      # 손바닥 중심 위치(base_link). orientation은 identity
bool detected                # 이 프레임에 손이 검출됨
bool fist                    # 주먹 확정(N프레임 연속 확인됨)
float32 confidence           # MediaPipe 검출 신뢰도
