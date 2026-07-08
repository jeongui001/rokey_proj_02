# servo: 파라미터 실기 튜닝 테스트 설계

## 배경

`robot_control_params.yaml`의 `servo:` 섹션(`kp_xy`, `v_max`, `descend_speed`,
`eps_descend`, `eps_grasp`, `n_stable`, `dt_latency`, `timeout`, `t_lost`,
`innov_low/high`, `w_alpha`, `z_close`, `diverge_n`, `cov_threshold`,
`kalman_q_pos/q_vel/r_xy/r_z`, `p0_vel_reset`)는 실기 튜닝 전까지 코드 기본값을
그대로 옮겨둔 상태다. `object_detection`(팀원3)과 `task_manager` 오케스트레이션은
아직 미완성이지만, 실물 M0609, `dsr_msgs2`(이 PC의 `~/cobot_ws`에 빌드됨),
RealSense, 학습된 YOLO weight(`vision_node/resource/tool_detector_best.pt`)는
모두 사용 가능한 상태다.

설계 과정에서 두 가지를 확인/수정했다:

1. **SpeedlStream 단위 불일치 버그 발견 및 수정**: `ServoLoop`는 m/s 단위로
   계산하는데(`task_executor.py`가 `GetCurrentPosx`의 mm 값을 `/1000.0`으로
   변환해서 넣어줌), `doosan_driver.py::publish_speedl`이 그 값을 변환 없이
   `SpeedlStream.vel`에 그대로 넣고 있었다. `SpeedlStream.acc`가 이미
   `speedl_acc_trans_mm_s2`(mm/s²)로 mm 기준이고, `tools/probe_speedl_stream.py`로
   `--vel-mm-s 20.0`이 적당한 속도임이 실기 확인된 바 있어, `SpeedlStream.vel`도
   mm/s가 맞다고 판단했다. `publish_speedl`에서 `command.vx/vy/vz`(m/s)에 1000을
   곱해 mm/s로 변환하도록 수정했고(`doosan_driver.py`), 회귀 테스트
   (`test_doosan_driver.py::test_publish_speedl_converts_mps_to_mmps`)를
   추가했다. 이 변환은 m 세계(비전/TF/서보 내부)와 mm 세계(두산 하드웨어 API)가
   만나는 두 지점 중 하나이며, 다른 하나는 `task_executor.py:284`의 읽기 방향
   변환이다. m을 mm으로 시스템 전체에서 통일하지 않은 이유는 아래 "단위 경계"
   절 참고.
2. **`servo_pick.hardware_ready` 게이트 재확인**: 이 파라미터는 ToolTrack
   orientation 의미, TCP/그리퍼 offset, SpeedlStream 단위 세 가지가 검증되기
   전까지 false로 유지하도록 설계돼 있다. SpeedlStream 단위는 위에서 해결됐고,
   orientation은 yaw 고정(1차 구현 범위 밖)이라 해당 없다. **TCP/그리퍼 offset은
   여전히 미검증**이므로, 이번 테스트는 그리퍼 폐합(CLOSING) 직전까지만 다루고
   실제 파지는 다음 라운드로 미룬다.

## 목적

1. `object_detection`/`task_manager` 완성을 기다리지 않고, 실물 M0609 +
   RealSense + 학습된 YOLO weight로 `servo_pick`의 TRACKING/DESCENDING 상태
   전환과 수렴 거동을 관찰하며 `servo:` 파라미터를 반복 튜닝할 수 있는 테스트
   경로를 만든다.
2. `should_abort()`의 각 사유(`timeout`, `tracking_lost`, `diverging`)가 의도한
   조건에서 정상적으로 트리거되는지 실기로 확인한다.

## 범위 밖

- `eps_grasp`, `z_close`, `n_stable`, `cov_threshold` 등 폐합(CLOSING) 판정
  파라미터의 최종 튜닝 — TCP/그리퍼 offset 실측 전이라 그리퍼 폐합까지 가지
  않는다(1차 라운드는 사람이 CLOSING 진입 전에 액션을 취소한다). offset 실측
  후 별도 라운드에서 다룬다.
- `task_manager`를 통한 전체 파이프라인 오케스트레이션(`SetVisionMode` 호출,
  재시도, FSM 전이 등) — `servo_pick` 액션을 직접 호출해 격리 테스트한다.
- `object_detection` 정식 패키지 구현 — 이 테스트에 필요한 최소 발행 노드만
  만든다(아래 컴포넌트 1).
- `handover_approach`, `handover_hold` 등 다른 액션.

## 단위 경계 (m ↔ mm)

시스템 전체를 mm으로 통일하지 않고 두산 하드웨어 경계에서만 변환하는 이유:
`KalmanXYZV`가 위치·속도를 같은 상태 벡터로 다루므로 `ToolTrack.pose.position`부터
mm이어야 필터 내부 속도도 mm/s가 되는데, `ToolTrack`은 `geometry_msgs/Pose`라
ROS 관례(REP 103)상 m이어야 하고 `vision_node`의 hand-eye TF(`camera_link`↔
`base_link`, `flange`)도 전부 m 기준이라 여기를 mm으로 바꾸면 TF 합성이 깨진다.

| 계열 | 단위 | 대표 파일 |
|---|---|---|
| 비전/TF/서보 내부 | m, m/s (분산은 m²) | `vision_node/tracking.py`, `vision_node/vision_node.py`, `handover_interfaces/msg/ToolTrack.msg`, `robot_control/kalman.py`, `robot_control/servo_loop.py`, `robot_control_params.yaml`의 `servo:` |
| 두산/그리퍼 하드웨어 경계 | mm, mm/s | `doosan_driver.py`(`GetCurrentPosx`, `SpeedlStream`), `tools/probe_speedl_stream.py`, `rg2.*_mm`, `task_manager_params.yaml`의 `driver.width_mm` |

변환 지점은 정확히 2곳: `task_executor.py:284`(mm→m, 읽기), `doosan_driver.py`의
`publish_speedl`(m→mm, 쓰기, 이번에 수정).

## 컴포넌트

1. **`vision_node/vision_node/tools/yolo_detection_publisher.py`** (신규)
   `fake_detection_publisher.py`와 같은 자리·역할이지만 가짜 bbox 대신
   `tool_detector_best.pt`(ultralytics YOLO)로 실제 추론한다.
   - `/camera/color/image_raw` 구독, 프레임마다 추론
   - 파라미터: `tool_class`(필터할 클래스명, 기본값 없이 필수 — 임의로 아무
     클래스나 잡지 않기 위함), `conf_threshold`(기본 0.5), `weight_path`(기본
     `tool_detector_best.pt`의 share 경로)
   - `tool_class`와 일치하는 detection 중 최고 score 1개만 `Detection2D`로 담아
     `DetectionArray`를 `/detection/tool_boxes`에 발행 (여러 개 감지돼도
     `vision_node.ToolTracker`가 이미 추적/매칭 로직을 갖고 있으므로 여기서는
     클래스 필터링만 하고 후보를 다 넘겨도 되지만, 최소 구현으로 top-1만 우선
     발행하고 필요시 확장)
   - 클래스 목록 불일치(`tool_class` 파라미터가 `model.names`에 없음) 시 노드
     시작을 거부(fail-closed) — 오탐으로 엉뚱한 물체를 쫓아가지 않도록

2. **`robot_control/robot_control/tools/trigger_servo_pick.py`** (신규)
   `task_manager` 없이 `servo_pick` 액션 goal을 직접 보내는 진단 스크립트
   (`probe_speedl_stream.py`와 같은 "실기 진단 도구" 계열).
   - CLI: `--robot-id`(dsr01), `--tool-class`(필수), `--grasp-width-mm`,
     `--grasp-force-n`(필수 — 추측 금지, 기존 관례와 동일)
   - `RobotTask.Feedback.state`를 매 틱 콘솔에 출력
   - Ctrl+C(SIGINT) 시 액션 `cancel_goal` 호출 후 종료 — 1차 라운드에서
     CLOSING 진입 전에 사람이 개입해 멈추는 수단

3. **`robot_control/robot_control/servo_loop.py` + `task_executor.py`에 디버그
   로그 추가** (신규, 작은 확장) — 현재 `RobotTask.Feedback.state`는
   TRACKING/DESCENDING 같은 상태 이름만 담고 있어 `kp_xy`/`innov_low/high` 같은
   파라미터를 튜닝하려면 `e_xy_norm`, `w`, `z_gap` 같은 수치가 필요하다.
   `safety.debug_log_tool_force`와 동일한 패턴으로 `servo_pick.debug_log_step`
   파라미터(기본 false)를 추가하고, true일 때만
   `_servo_pick_step`에서 throttle된(`throttle_duration_sec=0.2`) 로그로
   `e_xy_norm`, `w`, `state`, `z_gap`을 출력한다. `ServoLoop`에 이 값들을 읽을
   수 있는 최소 공개 속성(예: `last_error_xy`, `last_w`, `last_z_gap`)을 추가한다.

4. **`doosan_driver.py::publish_speedl` 단위 수정** — 이미 완료(위 배경 참고).

## 실행 절차 / 안전 설계

- launch: 기존 `vision_node.launch.py`(RealSense + hand-eye TF +
  `vision_node`) + `yolo_detection_publisher`(신규) + `robot_control_node`
  (`hardware_enabled:=true`)를 함께 띄우는 테스트 전용 launch 파일 또는 CLI
  조합으로 실행.
- `servo_pick.hardware_ready`는 yaml에 영구 기록하지 않고, 실행 시
  `-p servo_pick.hardware_ready:=true` 오버라이드로만 켠다.
- 최초 실행은 `v_max`, `descend_speed`를 yaml 기본값보다 낮춘 보수적 값(예:
  `v_max:=0.05`, `descend_speed:=0.03`)으로 오버라이드해서 시작하고, 거동을
  본 뒤 점진적으로 원복한다.
- 실행 전 확인 절차는 `probe_speedl_stream.py`와 동일: Enable 스위치/펜던트
  소지, 작업공간 확보, 로봇 주변 사람 접근 금지.
- 1차 라운드는 `should_close()`가 `True`가 되기 전에 `trigger_servo_pick.py`를
  Ctrl+C로 취소해 종료한다 — `eps_grasp`/`n_stable`/`cov_threshold` 등 폐합
  판정까지 실제로 실행되지 않게 사람이 개입한다.
- `SafetyMonitor`(외력 감지, DRFL 직접 연결)는 그대로 활성 상태로 둔다 — 별도
  안전판.

## 튜닝 순서 (낮은 리스크 → 높은 리스크)

1. **`kp_xy`, `v_max`**: 손으로 공구를 천천히 움직여 TCP가 따라오는지, 진동/
   발산 없이 수렴하는지 관찰 (`debug_log_step`의 `e_xy_norm` 추이 확인).
2. **`eps_descend`**: DESCENDING 전환 시점이 의도한 수평 오차에서 일어나는지.
3. **`dt_latency`, `innov_low/high`, `w_alpha`, `kalman_q_pos/q_vel/r_xy/r_z`**:
   `debug_log_step`의 `w` 값으로 필터 반응성/노이즈 저항 관찰 — 공구를 갑자기
   멈추거나 방향을 바꿔가며 `w`가 기대대로 0/1 사이를 오가는지 확인.
4. **`timeout`, `t_lost`, `diverge_n`**: 공구를 카메라 시야 밖으로 빼거나
   급격히 반대로 움직여 각 abort 사유(`timeout`/`tracking_lost`/`diverging`)가
   의도한 조건에서만 트리거되는지 확인.
5. **(범위 밖, 다음 라운드)** `eps_grasp`, `z_close`, `n_stable`,
   `cov_threshold` — TCP/그리퍼 offset 실측 후.

## 테스트 계획

- `yolo_detection_publisher.py`: bbox 선택(top-1, score 기준) 및
  `Detection2D` 변환 로직만 순수 함수로 분리해 pytest 검증. 실제 추론/카메라는
  수동 관찰 대상(자동 판정 없음, `probe_speedl_stream.py`와 동일 원칙).
- `trigger_servo_pick.py`: goal 메시지 구성 로직만 pytest 검증, 실행 자체는
  수동.
- `ServoLoop`의 신규 디버그 속성(`last_error_xy` 등)이 `step()` 이후 올바른
  값을 갖는지 `test_servo_loop.py`에 단위 테스트 추가.
- `doosan_driver.py` 단위 변환:
  `test_doosan_driver.py::test_publish_speedl_converts_mps_to_mmps` (완료,
  통과 확인됨).
- 실기 튜닝 세션 자체는 사람이 관찰해 판단하는 수동 절차이므로 자동 성공/실패
  기준은 없다.
