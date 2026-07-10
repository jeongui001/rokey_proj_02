string class_name
float32 score
int32 x1
int32 y1
int32 x2
int32 y2
# 2-keypoint pose (pose 모델일 때만 채워짐 - 공구 장축 양 끝점. 전환 계획:
# tools/KEYPOINT_LABELING_SPEC.md). conf<=0이면 keypoint 없음(구 box 모델)으로
# 취급하고 소비측(vision_node)은 bbox 중심/depth-PCA로 폴백한다.
float32 kpt0_x
float32 kpt0_y
float32 kpt0_conf
float32 kpt1_x
float32 kpt1_y
float32 kpt1_conf
