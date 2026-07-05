# 팀원2 구현 설계 — vision_tracking + calculate

## 배경과 범위

`docs/전체 계획.md` 8절 역할 분담에서 팀원2가 맡은 두 영역을 구현한다.

- `vision_node`: 공구/손 추적 (`_track_tool`, `_track_hand`)
- `robot_control`의 calculate 모듈(`servo_loop.py`): 칼만 필터·PBVS 제어법칙·피드포워드 가중치 `w`·파지판정·abort판정

기존 워크스페이스(`/home/hwangjeongui/rokey_proj_02/src`)에 이미 `vision_node.py`, `servo_loop.py`,
`robot_control_node.py` 스켈레톤이 있고, 채워야 할 부분이 `NotImplementedError`로 표시돼 있다.
이 설계는 그 스텁을 채우는 구체적인 알고리즘/인터페이스를 정의한다.

개발자 배경: ROS2 topic/service/action 작성 경험 있음, 칼만 필터·PID 구현 경험 있음 → 기초 설명 생략.
하드웨어(RealSense + Doosan 로봇팔) 즉시 접근 가능. YOLO 모델 학습은 팀원3 담당(범위 밖).

## A. 검출 인터페이스 (팀원3 ↔ 팀원2 계약)

`handover_interfaces`에 msg 2개 추가:

```
# msg/Detection2D.msg
string class_name
float32 score
int32 x1
int32 y1
int32 x2
int32 y2

# msg/DetectionArray.msg
std_msgs/Header header      # color 프레임과 동일 stamp
Detection2D[] detections
```

토픽 `/detection/tool_boxes` (`DetectionArray`)로 팀원3이 프레임마다 publish한다는 가정.
`vision_node`는 기존 color/depth/camera_info `message_filters` 싱크 그룹에 이 토픽도 추가한다
(slop 0.05 유지, 4-way `ApproximateTimeSynchronizer`).

팀원3 코드가 아직 없으므로, 이 토픽에 가짜 bbox를 publish하는 스크립트를 먼저 만들어
`vision_node`를 독립적으로 검증한다.

## B. vision_node `_track_tool`

입력: 동기화된 `(color_msg, depth_msg, info_msg, detection_msg)`, `tf_at_stamp`, `self.tool_class`.

1. `detection_msg.detections` 중 `class_name == self.tool_class`인 것만 후보로 필터링
2. 이전 프레임 추정 위치(내부 상태로 유지)가 있으면, 후보들의 bbox 중심을 3D로 복원해
   이전 위치와 가장 가까운 것을 선택(단순 최근접 매칭). 이전 추정이 없으면 최고 score 선택.
   후보가 없으면 이번 프레임은 스킵(발행 안 함)
3. bbox 중심 픽셀 + depth + `camera_info` intrinsics → camera 좌표 → `tf_at_stamp`로 base_link 변환
   (cobot_ws `detection.py`의 `_pixel_to_camera_coords` 패턴 재사용)
4. depth 무효(0 또는 MinZ 미만) → `depth_valid=false`. z는 마지막 유효값으로 고정하고,
   그 z값으로 픽셀→광선을 역산해 x·y는 계속 갱신
5. velocity는 이 단계에서는 가벼운 알파-베타 필터(포지션+속도, 게인 2개)로 추정한다.
   이 값은 접근판정·표시용이며, 실제 제어에 쓰이는 정밀 칼만 필터는 D절의 `servo_loop`가 별도로 갖는다
6. `approaching`: 속도벡터와 (기준점 − 현재위치) 벡터의 내적 부호로 판정.
   기준점은 파라미터(`vision.approach_ref_xy`)로 노출
7. yaw는 0 고정 (1차 구현 범위 밖으로 명시)

## C. vision_node `_track_hand`

MediaPipe Hands로 landmark 검출 → 손목 landmark 픽셀 선택 → depth 조회 →
B-3과 동일한 3D 복원 파이프라인 재사용 → `PoseStamped` 반환(orientation은 identity 고정).

## D. calculate 모듈 (`robot_control/servo_loop.py`)

- 상태 `[x, y, z, vx, vy]` 칼만 필터(등속 모델). `on_tool_track(msg)`에서 측정 갱신.
  `depth_valid=false`인 프레임은 측정 행렬 `H`에서 z 행을 제외한 부분 갱신(x,y만 갱신) —
  상태에 `vz`가 없으므로 z는 예측 단계에서 자연히 마지막 값이 유지된다
- innovation은 xy 잔차만 사용(노름). `innov >= innov_high` → 속도 공분산 리셋 + `w_target=0`.
  `innov <= innov_low` → `w_target=1`. 그 사이는 선형보간.
  `w = w_alpha * w_target + (1 - w_alpha) * w_prev`로 스무딩(EMA)
- `step(tcp_pose, now)`로 시그니처 변경(현재는 인자 없음). `tcp_pose`는 base_link 기준
  `(x,y,z,rx,ry,rz)`. 내부에서 `p_ref = p̂_tool(now + dt_latency + loop_half_period) + offset_approach`
  (`offset_approach`는 1차로 `(0,0)` 고정, 필요 시 파라미터화), `e = p_ref - p_tcp`,
  `v_cmd = clamp(w * v̂_tool + kp_xy * e, v_max)`. z(하강)/yaw는 2.3절 그대로. 반환값은 `ServoCommand`
- `robot_control_node.py`의 `_execute_servo_pick` 루프도 현재 tcp pose를 조회해 `step()`에
  넘기도록 수정한다. tcp pose를 실제로 읽는 부분(`_get_current_tcp_pose`)은 Doosan RT 연동이라
  팀원1 몫이므로 `NotImplementedError` 스텁으로만 추가
- `should_close`: `|e_xy| < eps_grasp`가 `n_stable` 연속 ∧ `z_gap < z_close`(2.6절) ∧ 필터 공분산 < 임계
- `should_abort`: 1차 구현은 아래 3개만 — 발산(`|e_xy|` 연속 증가), 추적유실(`t_lost` 초과), 타임아웃(`timeout_s` 초과).
  "방향전환 시야이탈 예상"은 판정 기준이 모호해 과설계 우려가 있으므로 1차 범위에서 제외하고
  코드 주석으로 명시(필요 시 추후 추가). 토크 이상은 `robot_control_node`가 별도로 처리(서보 루프 책임 아님)

## E. 테스트 하네스 (하드웨어 없이 튜닝)

- `ToolTrack`을 3가지 시나리오(①한방향 등속 ②긴주기 왕복 ③짧은주기 진동)로 흉내내 publish하는
  스크립트 (7절 검증 순서의 4번을 사전 시뮬레이션으로 미리 수행)
- 수신한 `v_cmd`를 적분해 다음 `tcp_pose`를 만들어주는 간단한 적분 시뮬레이터
- 위 둘로 `ServoLoop`를 하드웨어 없이 돌리며 `w`/게인을 1차 튜닝한 뒤, 실제 카메라·로봇으로 넘어간다

## 완료 기준

- `vision_node`, `servo_loop`의 기존 유닛테스트(`test_vision_node.py`, `test_servo_loop.py`)를
  새 동작에 맞게 갱신하고 통과
- 시뮬레이션 하네스로 3가지 시나리오에서 `w`가 기대대로 움직임을 확인(등속: `w≈1` 유지,
  방향전환: `w` 하강 후 재상승, 진동: `w` 감쇠)
- `handover_interfaces`에 추가한 msg가 빌드됨
