# 공구 전달 로봇 시스템 — 통신 구조 설계 스펙

기반 문서: `데모.md` (공구 전달 로봇 시스템 설계 문서)

## 0. 목적과 경계

이 스펙은 `데모.md`에 정의된 ROS2 멀티 패키지 시스템 중 **통신 구조(pub/sub/service/action 배선, 상태 전이 골격)** 만을 구현 대상으로 한다. 알고리즘(검출·추정·제어 로직)은 사용자가 직접 구현하므로 함수 시그니처 + TODO 스텁으로만 남긴다.

**TODO 스텁으로 남기는 알고리즘 목록:**
- `task_manager`: LLM 호출/파싱(`_call_llm`), 트리거 판정(`_check_trigger`), 삼중 확인 판정(`_verify_grasp`)
- `vision_node`: 공구 검출·3D복원·필터·approaching 판정(`_track_tool`), 손 검출(`_track_hand`)
- `robot_control`: PBVS 제어식·서보 루프 전체(`ServoLoop` 내부), RG2 Modbus 레지스터 제어(`RG2Client` 내부), Doosan 모션 서비스/RT 세션 호출(하드웨어 인터페이스 버전 미확정으로 TODO), fault 판정, 당김 감지 판정, 페이로드 추정
- `stt_node`: VAD(`_detect_voice_activity`), Whisper 추론(`_run_whisper`)

**제가 완전히 구현하는 부분:** 위 스텁들을 호출하는 통신 배선(구독/퍼블리시/서비스/액션 wiring), `task_manager`의 상태 전이 테이블 전체, `robot_control`의 액션 서버 오케스트레이션(RT 세션 open/close 순서, feedback 중계 등), 그리고 `handover_ui`(PyQt) 앱 전체.

## 1. 워크스페이스 구조

```
src/
├── handover_interfaces/   # ament_cmake (rosidl 생성 제약)
├── stt_node/              # ament_python
├── vision_node/           # ament_python
├── robot_control/         # ament_python
├── task_manager/          # ament_python
└── handover_ui/           # ament_python — PyQt5 + roslibpy
```

- ROS2 Humble, ament_python 기본 (interfaces만 ament_cmake)
- 각 패키지는 자기 launch 파일만 가짐. 전체 시스템을 한 번에 묶는 최상위 launch는 만들지 않음.
- `rosbridge_server`, `realsense2_camera`는 기성 패키지 — 소스 포함 안 하고 필요한 곳(`handover_ui`, `vision_node`)의 launch에서 include만 한다.
- UI는 브라우저가 아니라 PyQt: rosbridge(WebSocket 9090)에 `roslibpy`로 접속. `handover_ui` 패키지는 제가 완전히 구현한다.

### 하드웨어 연동 경계
Doosan 모션 서비스 호출, RT 세션 open/close, RG2 Modbus 레지스터 제어는 드라이버 버전에 따라 인터페이스가 달라지므로 실제 서비스 콜 코드를 작성하지 않고 **함수 시그니처 + 필요한 파라미터 + "어떤 드라이버 API를 찾아 채워야 하는지" 주석**만 남긴다.

## 2. `handover_interfaces`

데모.md 4절 스펙 그대로:
- `msg/ToolTrack.msg`
- `msg/GripperState.msg`
- `srv/SetVisionMode.srv`
- `action/RobotTask.action`

## 3. `stt_node`

- 퍼블리시: `/user_command/text` (`std_msgs/String`)
- 백그라운드 캡처 루프 골격 구현: `_on_utterance_ready(text)` 콜백이 호출되면 즉시 퍼블리시
- TODO: `_detect_voice_activity(audio_chunk) -> bool`, `_run_whisper(audio_segment) -> str`

## 4. `vision_node`

- 구독: `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `camera_info` (message_filters 동기화), TF
- 퍼블리시: `/vision/tool_track` (ToolTrack), `/vision/hand_pose` (PoseStamped)
- 서비스 서버: `/vision/set_mode` (SetVisionMode) — 모드/타겟 tool_class 상태 전환은 제가 구현
- 이미지 콜백에서 `header.stamp` 시각 TF 조회(`tf_buffer.lookup_transform(..., stamp)`)는 제가 구현
- TODO: `_track_tool(color, depth, tf_at_stamp, tool_class) -> ToolTrack|None`, `_track_hand(color) -> PoseStamped|None`
- 콜백은 현재 모드에 따라 스텁을 호출하고 반환값을 그대로 퍼블리시하는 배선만 담당

## 5. `robot_control`

```
robot_control/
├── robot_control_node.py   # RobotTask 액션 서버, /gripper/state, /robot/fault 발행
├── rg2_client.py           # RG2 Modbus TCP 클래스
└── servo_loop.py           # PBVS 서보 루프
```

액션 처리 흐름 (제가 구현하는 오케스트레이션):
- `move_named` / `move_pose` / `place_down`: goal 파싱 → `_call_move_service(...)`(TODO) → result
- `servo_pick`: RT 세션 open(TODO) → `/vision/tool_track` 직접 구독 시작 → 주기마다 `ServoLoop`에 트랙 전달 → `ServoLoop.get_state()`를 액션 Feedback으로 중계 → `ServoLoop.should_close()` True 시 `rg2_client.close(width, force)` 호출 → 페이로드 추정(TODO) → RT 세션 close → result. `ServoLoop.should_abort()`가 사유를 반환하면 abort 처리
- `handover_hold`: 컴플라이언스 on(TODO) → 로봇 상태 콜백마다 `_is_pull_detected(torque)`(TODO) 체크 → 감지 시 `rg2_client.open()` → result

상시 동작:
- 로봇 상태 구독 콜백에서 `_check_fault(state_msg)`(TODO) → 감지 시 `/robot/fault` 퍼블리시 + 진행 중 액션 abort
- 타이머로 `rg2_client.get_state()`(TODO) 주기 호출 → `/gripper/state` 상시 퍼블리시

`RG2Client`, `ServoLoop`는 메서드 시그니처 + docstring만 두고 내부 로직은 전부 TODO.

## 6. `task_manager`

상태: `IDLE → PARSING → MOVE_TO_WATCH → DETECT_TRACK → SERVO_PICK → VERIFY_GRASP → MOVE_SAFE → TRACK_HAND → WAIT_PULL → RELEASE → HOME → IDLE`, 모든 상태에서 `/robot/fault` 수신 시 `FAULT` (데모.md 3절 표 그대로).

제가 구현: 구독(`/user_command/text`, `/vision/tool_track`, `/robot/fault`), `/task/status` JSON 퍼블리시, `/vision/set_mode` 서비스 클라, `RobotTask` 액션 클라, 상태 전이 테이블 전체(재시도 카운터 포함, 임계값은 5절 파라미터를 ROS2 파라미터로 노출).

TODO: `_call_llm(text) -> dict`, `_check_trigger(tool_track_msg) -> bool`, `_verify_grasp(result) -> bool` (등록된 공구 스펙 데이터 소스는 사용자 자유 구현).

## 7. `handover_ui` (PyQt, 완전 구현)

```
handover_ui/
├── handover_ui/
│   ├── main.py            # QApplication 진입점
│   ├── main_window.py     # QMainWindow 레이아웃
│   └── ros_client.py      # roslibpy 래핑
└── package.xml, setup.py
```

- `roslibpy.Ros(host, port=9090)` 연결 + 연결 상태 표시 + 자동 재연결
- 구독 표시: `/task/status`(state/detail 라벨 + 이력 로그), `/gripper/state`(폭/grip_detected), `/robot/fault`(빨간 배너)
- 퍼블리시: `/user_command/text` (텍스트 입력 + 전송 버튼)
- PyQt5/roslibpy는 pip 의존성 — package.xml에 주석, README에 설치 안내

## 8. launch 및 검증

- 패키지별 자체 launch (`vision_node.launch.py`는 realsense2_camera include + hand-eye static TF, `handover_ui.launch.py`는 rosbridge_websocket include)
- 최상위 통합 launch는 없음

검증:
1. `colcon build` 성공 (인터페이스 코드 생성 포함)
2. 노드별 실행 후 `ros2 node info` / `ros2 topic list` / `ros2 service list` / `ros2 action list`로 4.5절 토픽 총괄표와 실제 배선 일치 확인
3. TODO 스텁이 더미 반환값이어도 액션 goal/feedback/result, 상태 전이가 끝까지 형식적으로 완주하는지 확인
4. rosbridge 구동 상태에서 PyQt UI 연결/구독/퍼블리시 동작 확인
