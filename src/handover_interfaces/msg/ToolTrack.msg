std_msgs/Header header
string tool_class
geometry_msgs/Pose pose
bool approaching
bool depth_valid
float32 confidence
# 선택된 검출의 YOLO pose keypoint 픽셀 좌표.
# pose 모델일 때만 채워지며 conf<=0이면 keypoint 없음(구 box 모델)으로 보면 된다.
float32 kpt0_x
float32 kpt0_y
float32 kpt0_conf
float32 kpt1_x
float32 kpt1_y
float32 kpt1_conf
