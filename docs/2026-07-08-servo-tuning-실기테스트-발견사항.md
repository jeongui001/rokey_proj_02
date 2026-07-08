# servo: 파라미터 실기 튜닝 시도 중 발견한 버그/이슈 기록

시도 목적: `robot_control_params.yaml`의 `servo:`(칼만 ServoLoop) 파라미터를 실기로
튜닝하려 했음. 실제로는 그 전에 파이프라인 전체(비전 포함)에 막혀있던 버그를
여러 개 발견/수정하게 됐고, 마지막에 vision_node(팀원 담당 영역)의 좌표 계산
자체를 검증해야 하는 지점까지 갔음 — 이건 이번 작업 범위를 벗어난다고 판단해
여기서 중단. 아래는 발견한 것들의 기록. **코드는 이 문서 작성 후 이 시점 이전으로
되돌림 - 아래 수정 내용은 코드에 없고 기록으로만 남음(단, 1번은 유지 여부 별도
확인 필요).**

## 발견/수정한 버그

### 1. `doosan_driver.py::publish_speedl` - m/s → mm/s 단위 변환 누락
`ServoLoop`는 m/s로 속도를 계산하는데, `publish_speedl`이 변환 없이
`SpeedlStream.vel`(mm/s, `tools/probe_speedl_stream.py`로 실기 확인된 단위)에
그대로 넣고 있었음. 실기에서 속도 명령이 의도한 값보다 1000배 느리게 전달되는
문제. `vx * 1000.0` 등으로 수정.
- **robot_control 쪽 버그, vision_node/캘리브레이션과 무관 - 유지할지 별도 판단.**

### 2. `vision_node.py`/`yolo_detection_publisher.py` - RealSense 토픽 이름 오류
`realsense2_camera`의 `rs_launch.py`는 `camera_name`/`camera_namespace` 기본값이
둘 다 `'camera'`라 실제 토픽은 `/camera/camera/color/image_raw` 등인데, 코드는
`/camera/color/image_raw`(camera 한 번)로 구독하고 있어서 이미지를 전혀 못 받고
있었음. `corecode/Calibration_Tutorial/realsense.py`(T_gripper2camera.npy를
뽑아낸 검증된 스크립트)의 토픽명과 대조해서 발견.

### 3. `vision_node.launch.py` - hand-eye 정적 TF의 부모 프레임 오류
static_transform_publisher가 `--frame-id flange`로 돼 있었는데, 실기 `/tf`를
직접 덤프해보니 M0609 URDF에는 `flange` 프레임이 없고 `base_link -> link_1 ->
... -> link_6`까지만 존재함. `flange`로 두면 로봇 트리와 연결 안 되는 별도
트리가 생겨 "two or more unconnected trees" 에러. `link_6`으로 수정.
- 캘리브레이션 수치(x/y/z, 쿼터니언) 자체는 안 건드림 - 프레임 이름만 실제
  URDF에 맞게 고친 것.

### 4. `vision_node.py::_track_tool` - ToolTrack.header.frame_id 오류
`track.header = color_msg.header`로 헤더를 통째로 복사해서 `frame_id`가 카메라
원본 프레임(`camera_color_optical_frame`)으로 남아있었음. position은 이미
`camera_to_base()`로 base_link 기준 변환을 마쳤는데 frame_id만 안 맞아서,
robot_control_node의 계약 검증(`servo_pick.tool_track_frame_id` 불일치 체크)에서
매 프레임 거부되고 있었음. stamp만 유지하고 frame_id는 `'base_link'`로 명시.

### 5. (버그는 아니고 성능) RealSense 프로파일 60fps → 30fps
이 개발 PC에 GPU가 없어(`torch.cuda.is_available()` False, `nvidia-smi` 없음)
YOLO 추론이 CPU로만 도는데, 60fps 입력을 못 따라가 시스템 전체가 렉 걸림.
30fps로 낮춤(`dt_latency` 0.05s보다는 짧은 33ms 유지).

## 미해결 - 여기서 중단한 이유

1~5를 다 고친 뒤 실기로 `servo_pick`을 태웠더니 **로봇이 렌치를 따라가지 않고
엉뚱한 곳으로 이동**함. 원인 후보:

> `robot_control_node.py`에 이미 명시된 경고: "ToolTrack.pose는 base_link
> 절대좌표로 정의되어 있는데, ServoLoop는 이를 TCP(그리퍼) 기준 xy 오차로
> 가정하고 P 제어를 수행한다. 이 좌표 변환이 실제로 구현·검증되기 전까지는
> 실제 속도 명령을 로봇에 보내면 안 된다." (`servo_pick.hardware_ready` 기본값
> false의 근거)

즉 hand-eye 캘리브레이션(`T_gripper2camera.npy`) 자체가 정확한지, 또는 로봇
컨트롤러에 설정된 RG2 그리퍼 TCP 오프셋이 실제 그리퍼 손끝과 일치하는지가
검증 안 된 상태. 이건 vision_node/캘리브레이션 담당 팀원 영역이라 이번
"servo 파라미터 튜닝" 목적 범위를 벗어난다고 판단해 여기서 중단함.

## 검증 방법 (필요 시 재개용)

로봇을 움직이지 않고 확인 가능:
1. `ros2 service call /dsr01/aux_control/get_current_posx dsr_msgs2/srv/GetCurrentPosx "{ref: 0}"`
   로 현재 TCP 위치(mm, DR_BASE) 조회
2. 렌치를 그 TCP 지점에 정확히 갖다 댐
3. `/vision/tool_track` 값(m, base_link)과 1번 값(÷1000)을 비교
4. 몇 cm 이내로 맞으면 캘리브레이션 정상, 크게 어긋나면 hand-eye 행렬/픽셀-카메라
   변환/TCP 오프셋 설정 쪽을 의심
