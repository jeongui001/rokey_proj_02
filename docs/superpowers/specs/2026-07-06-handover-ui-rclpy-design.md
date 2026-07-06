# handover_ui: rosbridge/roslibpy → rclpy 직접 통합 설계

## 배경

`handover_ui`는 원래 브라우저 UI로 계획되어 `rosbridge_server`(WebSocket) +
`roslibpy`로 ROS2와 통신하도록 설계됐다. 이후 UI 구현체는 PyQt 데스크톱 앱으로
바뀌었지만 통신 계층은 그대로 남아, 불필요한 WebSocket 중계 프로세스와 JSON
직렬화 왕복이 남아있다. PyQt는 파이썬 프로세스이므로 `rclpy`로 ROS2에 직접
붙을 수 있다. 이 문서는 그 전환의 설계다.

## 범위

- `handover_ui` 패키지만 변경한다. `task_manager`, `robot_control`,
  `vision_node`, `stt_node`는 토픽 이름·타입이 그대로이므로 로직 변경이
  필요 없다.
- `task_manager_node.py`, `robot_control_node.py`에 있는 "rosbridge 경유"
  주석 4곳은 실제 코드가 아니라 설명 주석이지만, rclpy 전환 후 사실과
  맞지 않게 되므로 문구만 정정한다(로직 변경 없음).
- `rosbridge_server`는 완전히 제거한다(선택적 유지 옵션 없음) — 결정 근거:
  handover_ui가 rosbridge의 유일한 소비자.

## 아키텍처

```
main.py
  rclpy.init()
  RosClient()  ── _HandoverUiNode(rclpy.Node) 생성
              └─ RosSpinThread(QThread) 생성 (아직 시작 안 함)
  MainWindow(ros_client)  ── ros_client.on_task_status 등 콜백 슬롯에
                              MainWindow의 pyqtSignal.emit을 꽂음
  ros_client.connect()   ── RosSpinThread.start() → rclpy.spin(node)
  ros_client.subscribe_all()
  app.exec_()
  (종료 시) app.aboutToQuit → ros_client.close() → rclpy.shutdown()
```

`main_window.py`는 **변경 없음**. 이미 "콜백이 다른 스레드에서 올 수 있다"는
전제로 pyqtSignal을 통해 위젯을 갱신하고 있어서(roslibpy 스레드 대응),
`RosClient`가 지금과 동일한 퍼블릭 인터페이스(`connect`, `close`,
`is_connected`, `subscribe_all`, `publish_command`, `on_task_status`,
`on_gripper_state`, `on_fault`)를 유지하는 한 그대로 재사용된다.

## 컴포넌트

### `_HandoverUiNode(rclpy.Node)`
- `/task/status`(String), `/gripper/state`(handover_interfaces/GripperState),
  `/robot/fault`(String) 구독. `/user_command/text`(String) 퍼블리셔.
- 구독 콜백은 파싱만 하고(예: `/task/status`의 JSON 디코딩), 결과는
  `RosSpinThread`의 pyqtSignal을 emit해 전달한다. 위젯을 직접 만지지 않는다.

### `RosSpinThread(QThread)`
- `task_status`, `gripper_state`, `fault` 3개의 `pyqtSignal` 보유.
- `run()`에서 `rclpy.spin(self.node)` 실행. `stop()`은
  `rclpy.shutdown()` 대신 `executor.shutdown()`/`spin` 루프 탈출 방식으로
  스레드를 정지시키고 `join`한다.

### `RosClient`
- `_HandoverUiNode` + `RosSpinThread`를 조립하는 얇은 래퍼. `__init__`에서
  스레드의 시그널을 각각 `self.on_task_status(...)` 등 기존 콜백 속성을
  호출하는 내부 슬롯에 connect한다.
- `connect()` = 스레드 시작, `close()` = 스레드 정지 + `node.destroy_node()`.
- `subscribe_all()`은 지금처럼 유지하되 내부적으로
  `node.create_subscription(...)` / 퍼블리셔 생성 호출로 바뀐다.

## 데이터 흐름 / 메시지 타입

| 토픽 | 타입 | 비고 |
|---|---|---|
| `/task/status` | `std_msgs/String` | 그대로 JSON 문자열 페이로드 — `json.loads(msg.data)`로 파싱, 인터페이스 변경 없음 |
| `/robot/fault` | `std_msgs/String` | `msg.data` 그대로 사용 |
| `/gripper/state` | `handover_interfaces/GripperState` | 이제 dict가 아니라 실제 메시지 객체 — `msg.width_mm`, `msg.grip_detected` 속성 직접 접근 |
| `/user_command/text` | `std_msgs/String` | `String(data=text)` 생성 후 publisher.publish |

## 생명주기 / 에러 처리

- `main.py` 호출 순서는 현재와 동일하게 유지: `RosClient()` 생성 →
  `MainWindow(ros_client)` 생성(콜백 슬롯 먼저 꽂기) → `connect()` →
  `subscribe_all()`. 앞에 `rclpy.init()` 한 줄만 추가된다.
- 현재 `main.py`는 앱 종료 시 정리 로직이 없다(roslibpy는 프로세스 종료로
  충분했음). rclpy는 정리 없이 죽으면 경고/좀비 스레드가 남을 수 있어
  `app.aboutToQuit.connect(...)`에 `ros_client.close()` +
  `rclpy.shutdown()` 호출을 새로 추가한다. 기존에 없던 기능이지만
  rclpy 전환에 필수로 딸려오는 보강으로 간주한다.

## 테스트

- `test_ros_client.py`는 전면 재작성한다. roslibpy 목킹 대신, 테스트 안에서
  별도의 임시 rclpy 노드를 만들어 실제로 퍼블리시하고
  `rclpy.spin_once(node, timeout_sec=...)`로 `_HandoverUiNode`의 콜백이
  불리는지 확인하는 표준 rclpy 테스트 패턴을 쓴다. `pytest` fixture로
  모듈 단위 `rclpy.init()`/`shutdown()`을 감싼다.
- `test_main_window.py`는 인터페이스가 그대로라 변경 없음.

## 건드릴 파일

| 파일 | 변경 |
|---|---|
| `handover_ui/ros_client.py` | 전면 재작성 |
| `handover_ui/main.py` | rclpy init/shutdown 추가 |
| `handover_ui/main_window.py` | 변경 없음 |
| `package.xml` | `rosbridge_server` exec_depend 제거, `rclpy`/`std_msgs`/`handover_interfaces`/`python3-pyqt5` depend 추가 |
| `requirements.txt` | `roslibpy` 제거 |
| `launch/handover_ui.launch.py` | rosbridge include 제거, `ExecuteProcess` → `launch_ros.actions.Node` |
| `test/test_ros_client.py` | 전면 재작성 |
| `test/test_main_window.py` | 변경 없음 |
| `task_manager/task_manager/task_manager_node.py` | 주석 문구만 정정("rosbridge 경유" → "rclpy 직접 구독"), 로직 변경 없음 |
| `robot_control/robot_control/robot_control_node.py` | 주석 문구만 정정, 로직 변경 없음 |
