# handover_ui: rosbridge → rclpy 전환 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `handover_ui`가 `rosbridge_server`+`roslibpy` 대신 `rclpy`로 ROS2에 직접 접속하도록 바꾼다.

**Architecture:** `ros_client.py`에 `_HandoverUiNode(rclpy.Node)`(실제 pub/sub) + `RosSpinThread(QThread)`(spin 루프 전용 워커) + `RosClient`(둘을 조립하는 얇은 래퍼, 기존과 동일한 퍼블릭 인터페이스 유지)를 둔다. `main_window.py`는 변경하지 않는다.

**Tech Stack:** ROS2 Humble, rclpy, PyQt5, pytest (colcon `ament_python` 패키지).

**Spec:** `docs/superpowers/specs/2026-07-06-handover-ui-rclpy-design.md`

## 검증 메모 (플랜 작성 중 실제로 dry-run해서 찾은 버그 2건)

플랜을 확정하기 전에 Task 1·2의 코드를 실제로 이 환경에 임시 적용해서 돌려봤다
(작업 트리는 검증 후 원상복구했다). 그 과정에서 초안 설계에는 없던 동시성
버그 2개를 발견해 Task 2의 구현 코드에 반영했다 — 아래 Task 2 Step 3의 코드가
이미 수정된 버전이다:

1. **`start()`/`stop()` 경쟁 상태**: `run()` 안에서 `self._running = True`를
   세팅하면, `connect()` 직후 바로 `close()`를 부르는 것처럼 두 호출이
   빠르게 연달아 오면 백그라운드 스레드가 `run()`에 진입하기도 전에
   메인 스레드의 `stop()`이 `_running=False`로 바꿔놓고, 뒤늦게 `run()`이
   그걸 다시 `True`로 덮어써 스레드가 영원히 도는 좀비 스레드가 된다.
   → `start()`를 오버라이드해 메인 스레드에서 미리 `True`로 세팅.
2. **전역 executor 충돌**: `rclpy.spin_once(node)`처럼 executor 인자
   없이 쓰면 프로세스 전역 executor(`get_global_executor()`)를 공유한다.
   이 스레드가 계속 그 전역 executor를 점유하는 동안 다른 스레드가 다른
   노드로 `rclpy.spin_once`를 호출하면 `ValueError: generator already
   executing`이 난다. → 노드 전용 `SingleThreadedExecutor`를 만들어 쓰면
   완전히 격리된다.

실제 프로덕션 코드(`main.py`)에서는 스레드가 하나뿐이라 이 두 버그가 겉으로
드러나지 않았을 수도 있지만(운이 좋으면 타이밍이 안 맞아 재현이 안 됨),
근본적으로 옳지 않은 코드라 반드시 고치고 가는 것이 맞다고 판단했다.

## Global Constraints

- 변경 범위는 `handover_ui` 패키지 + `task_manager_node.py`/`robot_control_node.py`의 주석 4곳뿐이다. 다른 패키지의 로직은 건드리지 않는다.
- `rosbridge_server`는 완전히 제거한다(옵션으로 남기지 않음).
- `main_window.py`, `test_main_window.py`는 변경하지 않는다(퍼블릭 인터페이스가 그대로이므로).
- `RosClient`의 퍼블릭 인터페이스(`connect`, `close`, `is_connected`, `subscribe_all`, `publish_command`, `on_task_status`, `on_gripper_state`, `on_fault`)는 시그니처를 유지한다.
- `/task/status`, `/robot/fault`는 그대로 `std_msgs/String`(JSON 페이로드), `/gripper/state`는 `handover_interfaces/GripperState` 실제 메시지 타입, `/user_command/text`는 `std_msgs/String`.
- 이 저장소에서 pytest를 돌릴 때는 ROS 환경을 먼저 source해야 한다: `source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash`. 이 샌드박스에는 pytest용 `anyio` 플러그인이 깨져 있어 `-p no:anyio`를 항상 붙여야 한다(이 프로젝트와 무관한 기존 환경 이슈).
- 알려진 별개 이슈: 이 샌드박스에서는 `test_main_window.py`(기존 파일, 이번 변경과 무관)가 `QMainWindow` 생성 시 Qt 플랫폼 통합 문제로 크래시한다. `_HandoverUiNode`/`RosClient` 테스트는 위젯을 만들지 않으므로 이 문제와 무관하고 정상 동작한다.

---

### Task 1: `_HandoverUiNode` — 구독 3개 파싱 + 퍼블리시

**Files:**
- Modify: `src/handover_ui/handover_ui/ros_client.py` (전체 교체 — 이 태스크에서는 `_HandoverUiNode`만 남긴다. `RosClient`/`RosSpinThread`는 Task 2에서 추가)
- Test: `src/handover_ui/test/test_ros_client.py` (전체 교체)

**Interfaces:**
- Produces: `_HandoverUiNode(owner)` — 생성자 인자 `owner`는 `on_task_status(state: str, detail: str)`, `on_gripper_state(width_mm: float, grip_detected: bool)`, `on_fault(message: str)` 속성(콜백 또는 `None`)을 가진 객체. 메서드: `subscribe_all()`, `publish_command(text: str)`.

- [ ] **Step 1: 기존 테스트 파일을 새 테스트로 전체 교체**

`src/handover_ui/test/test_ros_client.py`의 전체 내용을 아래로 바꾼다.

```python
import time

import pytest
import rclpy
from std_msgs.msg import String

from handover_interfaces.msg import GripperState
from handover_ui.ros_client import _HandoverUiNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _FakeOwner:
    def __init__(self):
        self.task_status_calls = []
        self.gripper_state_calls = []
        self.fault_calls = []
        self.on_task_status = lambda state, detail: self.task_status_calls.append((state, detail))
        self.on_gripper_state = lambda width, grip: self.gripper_state_calls.append((width, grip))
        self.on_fault = lambda msg: self.fault_calls.append(msg)


@pytest.fixture
def owner():
    return _FakeOwner()


@pytest.fixture
def node(owner):
    n = _HandoverUiNode(owner)
    n.subscribe_all()
    yield n
    n.destroy_node()


@pytest.fixture
def peer():
    p = rclpy.create_node('test_peer_node')
    yield p
    p.destroy_node()


def _spin_until(spin_target, predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(spin_target, timeout_sec=0.1)
        if predicate():
            return True
    return False


def test_task_status_callback_parses_json(node, owner, peer):
    pub = peer.create_publisher(String, '/task/status', 10)
    time.sleep(0.3)
    pub.publish(String(data='{"state": "IDLE", "detail": "ready"}'))

    assert _spin_until(node, lambda: owner.task_status_calls)
    assert owner.task_status_calls == [('IDLE', 'ready')]


def test_gripper_state_callback_forwards_fields(node, owner, peer):
    pub = peer.create_publisher(GripperState, '/gripper/state', 10)
    time.sleep(0.3)
    pub.publish(GripperState(width_mm=30.0, grip_detected=True))

    assert _spin_until(node, lambda: owner.gripper_state_calls)
    assert owner.gripper_state_calls == [(30.0, True)]


def test_fault_callback_forwards_message(node, owner, peer):
    pub = peer.create_publisher(String, '/robot/fault', 10)
    time.sleep(0.3)
    pub.publish(String(data='torque anomaly'))

    assert _spin_until(node, lambda: owner.fault_calls)
    assert owner.fault_calls == ['torque anomaly']


def test_publish_command_sends_message(node, peer):
    received = []
    peer.create_subscription(String, '/user_command/text', lambda m: received.append(m.data), 10)
    time.sleep(0.3)

    node.publish_command('스패너 갖다줘')

    assert _spin_until(peer, lambda: received)
    assert received == ['스패너 갖다줘']
```

- [ ] **Step 2: 테스트가 실패하는지 확인 (구현이 아직 없음)**

Run:
```bash
cd src/handover_ui
source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash
python3 -m pytest test/test_ros_client.py -q -p no:anyio
```
Expected: FAIL — `ImportError: cannot import name '_HandoverUiNode' from 'handover_ui.ros_client'` (기존 `ros_client.py`에는 `RosClient`만 있고 `_HandoverUiNode`가 없음)

- [ ] **Step 3: `ros_client.py`를 `_HandoverUiNode`만 남긴 내용으로 교체**

`src/handover_ui/handover_ui/ros_client.py`의 전체 내용을 아래로 바꾼다(이 시점에는 `RosClient`가 빠져 `main.py`가 일시적으로 깨지지만, Task 3에서 바로 고친다 — `main.py`/`main_window.py`에는 이 클래스를 직접 쓰는 단위 테스트가 없어 이번 태스크의 테스트에는 영향 없음).

```python
"""rclpy로 ROS2에 직접 접속해 handover_ui가 필요한 토픽을 구독/퍼블리시한다.

원래는 rosbridge_server(WebSocket) + roslibpy로 접속했으나(브라우저 UI 초안의
흔적), UI가 PyQt 데스크톱 앱으로 확정된 뒤에는 rclpy로 직접 붙는 것이 더
단순하다(docs/superpowers/specs/2026-07-06-handover-ui-rclpy-design.md 참고).
"""

import json

from rclpy.node import Node
from std_msgs.msg import String

from handover_interfaces.msg import GripperState


class _HandoverUiNode(Node):
    """실제 ROS2 통신을 담당하는 rclpy 노드.

    구독 콜백은 파싱만 하고, 결과 전달은 owner(RosClient)의 콜백 속성
    (on_task_status 등)을 직접 호출한다. owner가 그 자리에 Qt pyqtSignal의
    emit을 꽂아두므로(main_window.py), 콜백이 spin 스레드에서 불려도 Qt가
    알아서 메인 스레드로 큐잉해 위젯을 안전하게 갱신한다.
    """

    def __init__(self, owner):
        super().__init__('handover_ui')
        self._owner = owner
        self._command_pub = None

    def subscribe_all(self):
        self.create_subscription(String, '/task/status', self._on_task_status, 10)
        self.create_subscription(GripperState, '/gripper/state', self._on_gripper_state, 10)
        self.create_subscription(String, '/robot/fault', self._on_fault, 10)
        self._command_pub = self.create_publisher(String, '/user_command/text', 10)

    def publish_command(self, text: str):
        self._command_pub.publish(String(data=text))

    def _on_task_status(self, msg):
        if self._owner.on_task_status is None:
            return
        payload = json.loads(msg.data)
        self._owner.on_task_status(payload.get('state', ''), payload.get('detail', ''))

    def _on_gripper_state(self, msg):
        if self._owner.on_gripper_state is None:
            return
        self._owner.on_gripper_state(msg.width_mm, msg.grip_detected)

    def _on_fault(self, msg):
        if self._owner.on_fault is None:
            return
        self._owner.on_fault(msg.data)
```

- [ ] **Step 4: 테스트 통과 확인**

Run:
```bash
cd src/handover_ui
source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash
python3 -m pytest test/test_ros_client.py -q -p no:anyio
```
Expected: `4 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/handover_ui/handover_ui/ros_client.py src/handover_ui/test/test_ros_client.py
git commit -m "refactor(handover_ui): replace roslibpy node with rclpy _HandoverUiNode"
```

---

### Task 2: `RosSpinThread` + `RosClient` — 스레드 생명주기와 퍼블릭 인터페이스

**Files:**
- Modify: `src/handover_ui/handover_ui/ros_client.py` (Task 1 내용에 이어서 클래스 2개 추가)
- Test: `src/handover_ui/test/test_ros_client.py` (Task 1 내용에 이어서 테스트 추가)

**Interfaces:**
- Consumes: Task 1의 `_HandoverUiNode(owner)`.
- Produces: `RosClient()` — `connect()`, `close()`, `is_connected() -> bool`, `subscribe_all()`, `publish_command(text: str)`, 콜백 속성 `on_task_status`/`on_gripper_state`/`on_fault`(기본값 `None`). `RosSpinThread(node)` — `start()`(QThread 상속), `stop()`.

- [ ] **Step 1: 실패하는 테스트 추가**

`src/handover_ui/test/test_ros_client.py` 상단 import 줄을 바꾸고, 파일 맨 아래에 테스트를 추가한다.

import 줄 변경 (기존 한 줄을 아래로 교체):
```python
from handover_ui.ros_client import _HandoverUiNode
```
→
```python
from handover_ui.ros_client import RosClient, _HandoverUiNode
```

파일 맨 아래에 추가:
```python
@pytest.fixture
def client():
    c = RosClient()
    yield c
    c.close()


def test_connect_starts_and_close_stops_spin_thread(client):
    client.connect()
    assert client.is_connected() is True

    client.close()
    assert client.is_connected() is False


def test_subscribe_all_and_receive_task_status(client, peer):
    received = []
    client.on_task_status = lambda state, detail: received.append((state, detail))
    client.subscribe_all()
    client.connect()

    pub = peer.create_publisher(String, '/task/status', 10)
    time.sleep(0.3)
    pub.publish(String(data='{"state": "IDLE", "detail": "ready"}'))

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not received:
        time.sleep(0.05)

    assert received == [('IDLE', 'ready')]


def test_publish_command_via_client(client, peer):
    received = []
    peer.create_subscription(String, '/user_command/text', lambda m: received.append(m.data), 10)
    client.subscribe_all()
    client.connect()
    time.sleep(0.3)

    client.publish_command('스패너 갖다줘')

    assert _spin_until(peer, lambda: received)
    assert received == ['스패너 갖다줘']
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run:
```bash
cd src/handover_ui
source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash
QT_QPA_PLATFORM=offscreen python3 -m pytest test/test_ros_client.py -q -p no:anyio
```
Expected: FAIL — `ImportError: cannot import name 'RosClient' from 'handover_ui.ros_client'`

- [ ] **Step 3: `RosSpinThread`, `RosClient` 구현 추가**

`src/handover_ui/handover_ui/ros_client.py` 맨 위 import 블록을 아래로 교체한다(2줄 추가 — `SingleThreadedExecutor`를 쓰는 이유는 구현 코드 뒤 설명 참고):
```python
import json

import rclpy
from PyQt5.QtCore import QThread
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from handover_interfaces.msg import GripperState
```

파일 맨 아래(`_HandoverUiNode` 클래스 다음)에 추가:
```python
class RosSpinThread(QThread):
    """spin 루프만 도는 워커 스레드. 자체 시그널은 두지 않는다 - _HandoverUiNode가
    owner 콜백을 직접 호출하고, pyqtSignal.emit()은 호출 스레드와 무관하게
    수신측 스레드로 큐잉되므로 별도 중계가 필요 없다."""

    def __init__(self, node):
        super().__init__()
        self.node = node
        # 노드 전용 executor를 직접 들고 있는다 - rclpy.spin_once(node)처럼
        # 인자 없이 쓰면 프로세스 전역 executor(get_global_executor())를
        # 공유하는데, 이 스레드가 매 0.1s 그 전역 executor를 계속 점유하는
        # 동안 다른 스레드가 다른 노드로 rclpy.spin_once를 호출하면
        # "generator already executing" 예외가 난다(전역 executor가 동시에
        # 두 번 진입됨). 노드 전용 executor를 쓰면 이 스레드는 완전히
        # 격리되어 다른 스레드의 spin과 절대 부딪히지 않는다.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._running = False

    def start(self):
        # 메인 스레드에서 미리 True로 세팅해야 한다 - run() 안에서 세팅하면
        # "start() 직후 바로 stop()"처럼 빠르게 연달아 호출될 때 백그라운드
        # 스레드가 아직 run()에 진입하지 못한 사이 stop()이 _running=False로
        # 바꿔놔도 뒤늦게 run()이 그걸 다시 True로 덮어써 루프가 영원히 도는
        # 경쟁 상태(race)가 생긴다.
        self._running = True
        super().start()

    def run(self):
        while self._running and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def stop(self):
        self._running = False
        self.wait()
        self._executor.shutdown()


class RosClient:
    """handover_ui가 사용하는 통신 래퍼. MainWindow는 이 클래스의 콜백 슬롯
    (on_task_status/on_gripper_state/on_fault)에 자기 핸들러를 꽂아넣기만
    하면 되고, rclpy를 전혀 몰라도 된다(main_window.py 참고).
    """

    def __init__(self):
        self.on_task_status = None
        self.on_gripper_state = None
        self.on_fault = None
        self._node = _HandoverUiNode(self)
        self._spin_thread = RosSpinThread(self._node)

    def connect(self):
        self._spin_thread.start()

    def close(self):
        self._spin_thread.stop()
        self._node.destroy_node()

    def is_connected(self) -> bool:
        return self._spin_thread.isRunning()

    def subscribe_all(self):
        self._node.subscribe_all()

    def publish_command(self, text: str):
        self._node.publish_command(text)
```

- [ ] **Step 4: 테스트 통과 확인**

Run:
```bash
cd src/handover_ui
source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash
QT_QPA_PLATFORM=offscreen python3 -m pytest test/test_ros_client.py -q -p no:anyio
```
Expected: `7 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/handover_ui/handover_ui/ros_client.py src/handover_ui/test/test_ros_client.py
git commit -m "feat(handover_ui): add RosSpinThread/RosClient rclpy wiring"
```

---

### Task 3: `main.py` — rclpy 초기화/종료 배선

**Files:**
- Modify: `src/handover_ui/handover_ui/main.py`

**Interfaces:**
- Consumes: Task 2의 `RosClient()`(인자 없음), `RosClient.connect()`, `RosClient.close()`.

- [ ] **Step 1: `main.py` 전체 교체**

`src/handover_ui/handover_ui/main.py`의 전체 내용을 아래로 바꾼다.

```python
import sys

import rclpy
from PyQt5 import QtWidgets

from handover_ui.main_window import MainWindow
from handover_ui.ros_client import RosClient


def main():
    rclpy.init()
    app = QtWidgets.QApplication(sys.argv)
    ros_client = RosClient()
    # MainWindow 생성자가 ros_client.on_* 콜백을 먼저 꽂아두므로, 그 다음에
    # connect()/subscribe_all()을 호출해야 메시지를 놓치지 않는다.
    window = MainWindow(ros_client)
    ros_client.connect()
    ros_client.subscribe_all()
    window.show()

    def _shutdown():
        ros_client.close()
        rclpy.shutdown()

    app.aboutToQuit.connect(_shutdown)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: import가 깨지지 않는지 확인**

Run:
```bash
cd src/handover_ui
source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash
python3 -c "import handover_ui.main"
```
Expected: 아무 출력 없이 종료(exit code 0) — import 시점에는 `rclpy.init()`/`QApplication` 생성이 실행되지 않으므로(모듈 최상단이 아니라 `main()` 함수 안에 있음) 에러 없이 끝나야 한다.

- [ ] **Step 3: 커밋**

```bash
git add src/handover_ui/handover_ui/main.py
git commit -m "feat(handover_ui): wire rclpy init/shutdown lifecycle in main.py"
```

---

### Task 4: 패키지 설정 정리 — `package.xml`, `requirements.txt`, `setup.py`, launch

**Files:**
- Modify: `src/handover_ui/package.xml`
- Modify: `src/handover_ui/requirements.txt`
- Modify: `src/handover_ui/setup.py`
- Modify: `src/handover_ui/launch/handover_ui.launch.py`

**Interfaces:** 없음(설정 파일만).

- [ ] **Step 1: `package.xml` 수정**

`src/handover_ui/package.xml`의 `<description>`, `<exec_depend>` 줄을 찾아 아래처럼 바꾼다.

Before:
```xml
  <description>PyQt 데스크톱 UI - rosbridge(roslibpy) 경유로 /task/status, /gripper/state, /robot/fault 표시 및 /user_command/text 입력</description>
  <maintainer email="hwangjeongui01@gmail.com">hwangjeongui</maintainer>
  <license>MIT</license>

  <exec_depend>rosbridge_server</exec_depend>

  <!-- PyQt5, roslibpy, pytest-qt는 ROS 패키지가 아니라 pip 의존성.
       requirements.txt(런타임) / test-requirements.txt(테스트) 참고 -->
```

After:
```xml
  <description>PyQt 데스크톱 UI - rclpy로 ROS2에 직접 접속해 /task/status, /gripper/state, /robot/fault 표시 및 /user_command/text 입력</description>
  <maintainer email="hwangjeongui01@gmail.com">hwangjeongui</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>handover_interfaces</depend>

  <!-- PyQt5, pytest-qt는 ROS 패키지가 아니라 pip 의존성.
       requirements.txt(런타임) / test-requirements.txt(테스트) 참고 -->
```

- [ ] **Step 2: `requirements.txt`에서 `roslibpy` 제거**

`src/handover_ui/requirements.txt` 전체 내용을 아래로 바꾼다.

```
PyQt5
```

- [ ] **Step 3: `setup.py` description 수정**

`src/handover_ui/setup.py`에서 아래 한 줄을 바꾼다.

Before:
```python
    description='PyQt 데스크톱 UI (rosbridge 경유)',
```

After:
```python
    description='PyQt 데스크톱 UI (rclpy 직접 연동)',
```

- [ ] **Step 4: launch 파일에서 rosbridge 제거, UI를 실제 ROS2 노드로 실행**

`src/handover_ui/launch/handover_ui.launch.py`의 전체 내용을 아래로 바꾼다.

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ui_node = Node(package='handover_ui', executable='handover_ui', output='screen')
    return LaunchDescription([ui_node])
```

- [ ] **Step 5: 문법 확인**

Run:
```bash
cd src/handover_ui
source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash
python3 -c "import launch_ros.actions; exec(open('launch/handover_ui.launch.py').read()); print(generate_launch_description())"
```
Expected: `LaunchDescription` 객체가 에러 없이 출력됨(리스트 안에 `Node` 액션 1개).

- [ ] **Step 6: 커밋**

```bash
git add src/handover_ui/package.xml src/handover_ui/requirements.txt src/handover_ui/setup.py src/handover_ui/launch/handover_ui.launch.py
git commit -m "chore(handover_ui): drop rosbridge_server/roslibpy from packaging and launch"
```

---

### Task 5: 다른 패키지의 "rosbridge 경유" 주석 정정

**Files:**
- Modify: `src/task_manager/task_manager/task_manager_node.py:55,57`
- Modify: `src/robot_control/robot_control/robot_control_node.py:65,66`

**Interfaces:** 없음(주석뿐, 로직 변경 없음).

- [ ] **Step 1: `task_manager_node.py` 주석 정정**

`src/task_manager/task_manager/task_manager_node.py`에서:

Before (55행):
```python
        self.pub_status = self.create_publisher(String, '/task/status', 10)  # 서브스크라이버: handover_ui(rosbridge 경유)
```
After:
```python
        self.pub_status = self.create_publisher(String, '/task/status', 10)  # 서브스크라이버: handover_ui(rclpy 직접 구독)
```

Before (57행):
```python
            String, '/user_command/text', self._on_user_command, 10)  # 퍼블리셔: stt_node, handover_ui(rosbridge 경유)
```
After:
```python
            String, '/user_command/text', self._on_user_command, 10)  # 퍼블리셔: stt_node, handover_ui(rclpy 직접 구독)
```

- [ ] **Step 2: `robot_control_node.py` 주석 정정**

`src/robot_control/robot_control/robot_control_node.py`에서:

Before (65행):
```python
        self.pub_gripper_state = self.create_publisher(GripperState, '/gripper/state', 10)  # 서브스크라이버: handover_ui(rosbridge 경유)
```
After:
```python
        self.pub_gripper_state = self.create_publisher(GripperState, '/gripper/state', 10)  # 서브스크라이버: handover_ui(rclpy 직접 구독)
```

Before (66행):
```python
        self.pub_fault = self.create_publisher(String, '/robot/fault', 10)  # 서브스크라이버: task_manager, handover_ui(rosbridge 경유)
```
After:
```python
        self.pub_fault = self.create_publisher(String, '/robot/fault', 10)  # 서브스크라이버: task_manager, handover_ui(rclpy 직접 구독)
```

- [ ] **Step 3: 기존 테스트 회귀 확인 (주석뿐이라 로직 영향 없어야 함)**

Run:
```bash
source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash
cd src/task_manager && python3 -m pytest test/ -q -p no:anyio && cd ../robot_control && python3 -m pytest test/ -q -p no:anyio
```
Expected: 두 패키지 모두 기존과 동일하게 전부 `passed`(주석만 바꿨으므로 결과 변화 없음).

- [ ] **Step 4: 커밋**

```bash
git add src/task_manager/task_manager/task_manager_node.py src/robot_control/robot_control/robot_control_node.py
git commit -m "docs(comments): correct handover_ui subscriber comment after rclpy migration"
```

---

### Task 6: 전체 검증

**Files:** 없음(검증만).

- [ ] **Step 1: `handover_ui` 전체 테스트 스위트 실행**

Run:
```bash
cd src/handover_ui
source /opt/ros/humble/setup.bash && source /home/hwangjeongui/rokey_proj_02/install/setup.bash
QT_QPA_PLATFORM=offscreen python3 -m pytest test/test_ros_client.py -q -p no:anyio
```
Expected: `7 passed`.

`test_main_window.py`는 이 샌드박스에서 `QMainWindow` 생성 시 기존부터(이번 변경과 무관하게) 크래시하는 환경 이슈가 있다(Global Constraints 참고). 실제 디스플레이가 있는 개발 환경에서 `python3 -m pytest test/test_main_window.py -q -p no:anyio`를 돌려 5개 테스트가 전부 통과하는지 별도로 확인한다 — 이번 변경으로 `main_window.py`를 건드리지 않았으므로 회귀가 없어야 한다.

- [ ] **Step 2: rosbridge/roslibpy 잔재 제거 확인**

Run:
```bash
grep -rn "roslibpy\|rosbridge" /home/hwangjeongui/rokey_proj_02/src/handover_ui /home/hwangjeongui/rokey_proj_02/src/task_manager/task_manager/task_manager_node.py /home/hwangjeongui/rokey_proj_02/src/robot_control/robot_control/robot_control_node.py
```
Expected: 아무 결과 없음(exit code 1, no matches).

- [ ] **Step 3: 최종 상태 확인**

Run: `git status`
Expected: 변경사항이 모두 커밋되어 클린 상태(untracked/modified 없음).
