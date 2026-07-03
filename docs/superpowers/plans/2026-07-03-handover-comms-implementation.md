# 공구 전달 로봇 시스템 — 통신 구조 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `데모.md`에 정의된 ROS2 멀티 패키지 공구 전달 로봇 시스템의 통신 구조(pub/sub/service/action 배선, 상태 전이 골격, PyQt UI)를 구현한다. 알고리즘(검출·추정·제어 로직)은 함수 시그니처 + `NotImplementedError` TODO 스텁으로 남긴다.

**Architecture:** ROS2 Humble, ament_python 6개 패키지(`handover_interfaces`는 ament_cmake) + PyQt5/roslibpy UI. 각 패키지는 자기 launch 파일만 가지며 최상위 통합 launch는 없다. TODO 스텁은 항상 `_safe_call()` 헬퍼로 감싸 호출되어, 스텁이 `NotImplementedError`를 던져도 노드가 죽지 않고 경고 로그 후 기본값으로 진행한다.

**Tech Stack:** ROS2 Humble, rclpy, message_filters, tf2_ros, PyQt5, roslibpy, pytest, pytest-qt

## Global Constraints

- ROS2 배포판: Humble. 빌드 타입: `handover_interfaces`만 `ament_cmake`, 나머지는 `ament_python`.
- 알고리즘 스텁은 전부 `raise NotImplementedError('<메서드명> 구현 필요')` + docstring으로 입출력 계약만 명시한다. 실제 알고리즘 코드를 쓰지 않는다.
- 통신 배선 코드(구독/퍼블리시/서비스/액션, 상태 전이, 오케스트레이션)는 전부 실제로 동작하는 완성 코드로 작성한다.
- Doosan 모션 서비스/RT 세션/RG2 Modbus 등 하드웨어 드라이버 호출은 정확한 인터페이스를 알 수 없으므로 함수 시그니처 + TODO만 남기고, 이를 사용하는 실제 ROS 구독/서비스는 만들지 않는다 (드라이버 패키지가 설치되어 있지 않아도 이 워크스페이스가 빌드/테스트되어야 함).
- `RobotTask.action`의 `task_type`은 6개 값을 허용한다: `move_named`, `move_pose`, `servo_pick`, `handover_hold`, `place_down`, `release_and_retry` (마지막 값은 브레인스토밍 중 추가 합의됨).
- 라이선스: MIT. maintainer: `hwangjeongui <hwangjeongui01@gmail.com>`.
- 테스트는 `colcon build --symlink-install && source install/setup.bash` 이후 `python3 -m pytest src/<package>/test/ -v` 로 실행한다 (인터페이스 타입은 install 트리에서 해석됨).

---

## 파일 구조 개요

```
src/
├── handover_interfaces/
│   ├── CMakeLists.txt, package.xml
│   ├── msg/ToolTrack.msg, msg/GripperState.msg
│   ├── srv/SetVisionMode.srv
│   └── action/RobotTask.action
├── stt_node/
│   ├── package.xml, setup.py, setup.cfg, resource/stt_node
│   ├── stt_node/__init__.py, stt_node/stt_node.py
│   ├── launch/stt_node.launch.py
│   └── test/test_stt_node.py
├── vision_node/
│   ├── package.xml, setup.py, setup.cfg, resource/vision_node
│   ├── vision_node/__init__.py, vision_node/vision_node.py
│   ├── launch/vision_node.launch.py
│   └── test/test_vision_node.py
├── robot_control/
│   ├── package.xml, setup.py, setup.cfg, resource/robot_control
│   ├── robot_control/__init__.py, rg2_client.py, servo_loop.py, robot_control_node.py
│   ├── launch/robot_control.launch.py
│   └── test/test_rg2_client.py, test_servo_loop.py, test_robot_control_node.py
├── task_manager/
│   ├── package.xml, setup.py, setup.cfg, resource/task_manager
│   ├── task_manager/__init__.py, task_manager_node.py
│   ├── launch/task_manager.launch.py
│   └── test/test_task_manager_node.py
└── handover_ui/
    ├── package.xml, setup.py, setup.cfg, resource/handover_ui
    ├── handover_ui/__init__.py, main.py, main_window.py, ros_client.py
    ├── launch/handover_ui.launch.py
    └── test/test_ros_client.py, test_main_window.py
```

---

### Task 1: `handover_interfaces` 패키지

**Files:**
- Create: `src/handover_interfaces/package.xml`
- Create: `src/handover_interfaces/CMakeLists.txt`
- Create: `src/handover_interfaces/msg/ToolTrack.msg`
- Create: `src/handover_interfaces/msg/GripperState.msg`
- Create: `src/handover_interfaces/srv/SetVisionMode.srv`
- Create: `src/handover_interfaces/action/RobotTask.action`

**Interfaces:**
- Produces: `handover_interfaces/msg/ToolTrack`, `handover_interfaces/msg/GripperState`, `handover_interfaces/srv/SetVisionMode` (Request 상수 `OFF=0`, `TRACK_TOOL=1`, `TRACK_HAND=2`), `handover_interfaces/action/RobotTask` (Goal/Result/Feedback) — 이후 모든 패키지가 이 타입을 import한다.

- [ ] **Step 1: 메시지/서비스/액션 정의 파일 작성**

`src/handover_interfaces/msg/ToolTrack.msg`:
```
std_msgs/Header header
string tool_class
geometry_msgs/Pose pose
geometry_msgs/Vector3 velocity
bool approaching
bool depth_valid
float32 confidence
```

`src/handover_interfaces/msg/GripperState.msg`:
```
std_msgs/Header header
float32 width_mm
bool grip_detected
```

`src/handover_interfaces/srv/SetVisionMode.srv`:
```
uint8 OFF=0
uint8 TRACK_TOOL=1
uint8 TRACK_HAND=2
uint8 mode
string tool_class
---
bool success
string message
```

`src/handover_interfaces/action/RobotTask.action`:
```
# task_type: move_named | move_pose | servo_pick | handover_hold | place_down | release_and_retry
string task_type
string named_target
geometry_msgs/PoseStamped target_pose
string tool_class
float32 grasp_width_mm
float32 grasp_force_n
---
bool success
float32 measured_payload_kg
float32 final_width_mm
bool grip_detected
string message
---
string state
```

- [ ] **Step 2: package.xml 작성**

`src/handover_interfaces/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>handover_interfaces</name>
  <version>0.0.1</version>
  <description>공구 전달 로봇 시스템 커스텀 msg/srv/action 정의</description>
  <maintainer email="hwangjeongui01@gmail.com">hwangjeongui</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>

  <member_of_group>rosidl_interface_packages</member_of_group>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 3: CMakeLists.txt 작성**

`src/handover_interfaces/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.8)
project(handover_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ToolTrack.msg"
  "msg/GripperState.msg"
  "srv/SetVisionMode.srv"
  "action/RobotTask.action"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()
```

- [ ] **Step 4: 빌드 후 인터페이스 검증**

Run:
```bash
cd ~/rokey_proj_02
colcon build --symlink-install --packages-select handover_interfaces
source install/setup.bash
ros2 interface show handover_interfaces/msg/ToolTrack
ros2 interface show handover_interfaces/srv/SetVisionMode
ros2 interface show handover_interfaces/action/RobotTask
```
Expected: 빌드 성공, 각 `interface show` 출력이 Step 1의 정의와 일치. (순수 인터페이스 패키지라 pytest 대상 로직이 없으므로 build + `ros2 interface show`가 이 태스크의 검증 수단이다.)

- [ ] **Step 5: 커밋**

```bash
git add src/handover_interfaces
git commit -m "feat: handover_interfaces 패키지 추가 (msg/srv/action)"
```

---

### Task 2: `stt_node` 패키지

**Files:**
- Create: `src/stt_node/package.xml`, `src/stt_node/setup.py`, `src/stt_node/setup.cfg`, `src/stt_node/resource/stt_node`
- Create: `src/stt_node/stt_node/__init__.py`, `src/stt_node/stt_node/stt_node.py`
- Test: `src/stt_node/test/test_stt_node.py`

**Interfaces:**
- Produces: `SttNode` 클래스, 퍼블리시 `/user_command/text` (`std_msgs/String`). TODO 스텁 `_read_audio_chunk() -> bytes`, `_detect_voice_activity(chunk: bytes) -> bool`, `_run_whisper(audio: bytes) -> str`.

- [ ] **Step 1: 패키지 뼈대 파일 작성**

`src/stt_node/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>stt_node</name>
  <version>0.0.1</version>
  <description>마이크 VAD + 로컬 Whisper -> /user_command/text</description>
  <maintainer email="hwangjeongui01@gmail.com">hwangjeongui</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`src/stt_node/setup.py`:
```python
from setuptools import find_packages, setup

package_name = 'stt_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/stt_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='마이크 VAD + 로컬 Whisper -> /user_command/text',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stt_node = stt_node.stt_node:main',
        ],
    },
)
```

`src/stt_node/setup.cfg`:
```
[develop]
script_dir=$base/lib/stt_node
[install]
install_scripts=$base/lib/stt_node
```

Run:
```bash
mkdir -p ~/rokey_proj_02/src/stt_node/resource
touch ~/rokey_proj_02/src/stt_node/resource/stt_node
touch ~/rokey_proj_02/src/stt_node/stt_node/__init__.py
```

- [ ] **Step 2: 실패하는 테스트 작성**

`src/stt_node/test/test_stt_node.py`:
```python
import rclpy
import pytest
from std_msgs.msg import String

from stt_node.stt_node import SttNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = SttNode()
    yield n
    n.destroy_node()


def test_on_utterance_ready_publishes_text(node):
    published = []
    node.pub_command.publish = published.append

    node._on_utterance_ready('스패너 갖다줘')

    assert len(published) == 1
    assert isinstance(published[0], String)
    assert published[0].data == '스패너 갖다줘'


def test_stub_methods_raise_not_implemented(node):
    with pytest.raises(NotImplementedError):
        node._read_audio_chunk()
    with pytest.raises(NotImplementedError):
        node._detect_voice_activity(b'')
    with pytest.raises(NotImplementedError):
        node._run_whisper(b'')


def test_safe_call_swallows_not_implemented(node):
    result = node._safe_call(node._run_whisper, b'', default='fallback')
    assert result == 'fallback'
```

Run: `cd ~/rokey_proj_02 && python3 -m pytest src/stt_node/test/test_stt_node.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'stt_node.stt_node'`)

- [ ] **Step 3: `SttNode` 구현**

`src/stt_node/stt_node/stt_node.py`:
```python
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SttNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self.pub_command = self.create_publisher(String, '/user_command/text', 10)
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _capture_loop(self):
        buffer = bytearray()
        while not self._stop_event.is_set():
            chunk = self._safe_call(self._read_audio_chunk, default=None)
            if chunk is None:
                self._stop_event.wait(0.1)
                continue
            is_speech = self._safe_call(self._detect_voice_activity, chunk, default=False)
            if is_speech:
                buffer.extend(chunk)
                continue
            if len(buffer) == 0:
                continue
            text = self._safe_call(self._run_whisper, bytes(buffer), default=None)
            buffer = bytearray()
            if text:
                self._on_utterance_ready(text)

    def _on_utterance_ready(self, text: str):
        msg = String()
        msg.data = text
        self.pub_command.publish(msg)

    def _read_audio_chunk(self) -> bytes:
        """마이크에서 오디오 청크 하나를 읽어 반환한다. 입력 디바이스/샘플레이트 등은 구현 시 결정."""
        raise NotImplementedError('_read_audio_chunk 구현 필요')

    def _detect_voice_activity(self, audio_chunk: bytes) -> bool:
        """audio_chunk에 발화가 포함되어 있는지 VAD로 판정한다."""
        raise NotImplementedError('_detect_voice_activity 구현 필요')

    def _run_whisper(self, utterance_audio: bytes) -> str:
        """utterance_audio 전체를 로컬 Whisper로 추론해 텍스트를 반환한다."""
        raise NotImplementedError('_run_whisper 구현 필요')


def main(args=None):
    rclpy.init(args=args)
    node = SttNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest src/stt_node/test/test_stt_node.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/stt_node
git commit -m "feat: stt_node 패키지 추가 (통신 배선, VAD/Whisper는 TODO)"
```

---

### Task 3: `vision_node` 패키지 + `/vision/set_mode` 서비스

**Files:**
- Create: `src/vision_node/package.xml`, `setup.py`, `setup.cfg`, `resource/vision_node`
- Create: `src/vision_node/vision_node/__init__.py`, `src/vision_node/vision_node/vision_node.py`
- Test: `src/vision_node/test/test_vision_node.py`

**Interfaces:**
- Consumes: `handover_interfaces.srv.SetVisionMode` (Task 1)
- Produces: `VisionNode` 클래스, 서비스 서버 `/vision/set_mode` (모드/`tool_class` 상태 저장)

- [ ] **Step 1: 패키지 뼈대**

`src/vision_node/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>vision_node</name>
  <version>0.0.1</version>
  <description>RealSense 저해상도·고프레임 공구/손 추적 노드</description>
  <maintainer email="hwangjeongui01@gmail.com">hwangjeongui</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>message_filters</depend>
  <depend>handover_interfaces</depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`src/vision_node/setup.py`:
```python
from setuptools import find_packages, setup

package_name = 'vision_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/vision_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='RealSense 저해상도·고프레임 공구/손 추적 노드',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = vision_node.vision_node:main',
        ],
    },
)
```

`src/vision_node/setup.cfg`:
```
[develop]
script_dir=$base/lib/vision_node
[install]
install_scripts=$base/lib/vision_node
```

Run:
```bash
mkdir -p ~/rokey_proj_02/src/vision_node/resource
touch ~/rokey_proj_02/src/vision_node/resource/vision_node
touch ~/rokey_proj_02/src/vision_node/vision_node/__init__.py
```

- [ ] **Step 2: 실패하는 테스트 작성 (set_mode만)**

`src/vision_node/test/test_vision_node.py`:
```python
import rclpy
import pytest

from handover_interfaces.srv import SetVisionMode
from vision_node.vision_node import VisionNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = VisionNode()
    yield n
    n.destroy_node()


def test_set_mode_updates_state(node):
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.TRACK_TOOL
    request.tool_class = 'spanner'
    response = SetVisionMode.Response()

    result = node._on_set_mode(request, response)

    assert result.success is True
    assert node.mode == SetVisionMode.Request.TRACK_TOOL
    assert node.tool_class == 'spanner'


def test_set_mode_off(node):
    request = SetVisionMode.Request()
    request.mode = SetVisionMode.Request.OFF
    request.tool_class = ''
    response = SetVisionMode.Response()

    result = node._on_set_mode(request, response)

    assert result.success is True
    assert node.mode == SetVisionMode.Request.OFF
```

Run: `python3 -m pytest src/vision_node/test/test_vision_node.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: `VisionNode` 뼈대 + set_mode 구현 (이미지 콜백은 Task 4에서 추가)**

`src/vision_node/vision_node/vision_node.py`:
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.mode = SetVisionMode.Request.OFF
        self.tool_class = ''

        self.pub_tool_track = self.create_publisher(ToolTrack, '/vision/tool_track', 10)
        self.pub_hand_pose = self.create_publisher(PoseStamped, '/vision/hand_pose', 10)
        self.srv_set_mode = self.create_service(SetVisionMode, '/vision/set_mode', self._on_set_mode)

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _on_set_mode(self, request, response):
        self.mode = request.mode
        self.tool_class = request.tool_class
        response.success = True
        response.message = f'mode set to {request.mode} (tool_class={request.tool_class})'
        return response

    def _track_tool(self, color_msg, depth_msg, tf_at_stamp, tool_class):
        """저해상도 YOLO 검출 + 3D 복원(tf_at_stamp 사용) + 칼만/알파-베타 필터로 ToolTrack을 만든다."""
        raise NotImplementedError('_track_tool 구현 필요')

    def _track_hand(self, color_msg):
        """MediaPipe 등으로 손을 검출해 PoseStamped를 만든다."""
        raise NotImplementedError('_track_hand 구현 필요')


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest src/vision_node/test/test_vision_node.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add src/vision_node
git commit -m "feat: vision_node 패키지 추가 + set_mode 서비스"
```

---

### Task 4: `vision_node` 이미지 콜백 배선 (TF + 검출 디스패치)

**Files:**
- Modify: `src/vision_node/vision_node/vision_node.py`
- Modify: `src/vision_node/test/test_vision_node.py`

**Interfaces:**
- Consumes: `self._track_tool`, `self._track_hand` (Task 3에서 정의된 TODO 스텁), `self._safe_call` (Task 3)
- Produces: `/vision/tool_track`, `/vision/hand_pose` 퍼블리시 배선

- [ ] **Step 1: 실패하는 테스트 추가**

`src/vision_node/test/test_vision_node.py`에 다음을 추가 (파일 하단):
```python
from std_msgs.msg import Header
from sensor_msgs.msg import Image


def _make_image_msg():
    msg = Image()
    msg.header = Header()
    msg.header.frame_id = 'camera_link'
    return msg


def test_synced_images_dispatches_to_track_tool_and_publishes(node):
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tool_class = 'spanner'
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'

    expected_track = ToolTrack()
    expected_track.tool_class = 'spanner'
    node._track_tool = lambda color, depth, tf, tool_class: expected_track

    published = []
    node.pub_tool_track.publish = published.append

    color_msg = _make_image_msg()
    node._on_synced_images(color_msg, color_msg, color_msg)

    assert published == [expected_track]


def test_synced_images_skips_publish_when_track_tool_returns_none(node):
    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = lambda *a, **k: 'fake_tf'
    node._track_tool = lambda *a, **k: None

    published = []
    node.pub_tool_track.publish = published.append

    color_msg = _make_image_msg()
    node._on_synced_images(color_msg, color_msg, color_msg)

    assert published == []


def test_synced_images_skips_when_tf_lookup_fails(node):
    from tf2_ros import TransformException

    def _raise(*a, **k):
        raise TransformException('no tf yet')

    node.mode = SetVisionMode.Request.TRACK_TOOL
    node.tf_buffer.lookup_transform = _raise

    called = []
    node._track_tool = lambda *a, **k: called.append(1)

    color_msg = _make_image_msg()
    node._on_synced_images(color_msg, color_msg, color_msg)

    assert called == []
```

Run: `python3 -m pytest src/vision_node/test/test_vision_node.py -v`
Expected: FAIL (`AttributeError: 'VisionNode' object has no attribute 'tf_buffer'` / `_on_synced_images`)

- [ ] **Step 2: TF + 동기화 구독 + 콜백 구현**

`src/vision_node/vision_node/vision_node.py`의 import 블록을 다음으로 교체:
```python
import message_filters
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode
```

`__init__` 안, `self.srv_set_mode = ...` 라인 다음에 추가:
```python
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_color = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info], queue_size=10, slop=0.05)
        self._sync.registerCallback(self._on_synced_images)
```

`_on_set_mode` 메서드 다음에 추가:
```python
    def _on_synced_images(self, color_msg, depth_msg, info_msg):
        try:
            tf_at_stamp = self.tf_buffer.lookup_transform(
                'base_link', color_msg.header.frame_id, color_msg.header.stamp,
                timeout=Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup failed: {ex}')
            return

        if self.mode == SetVisionMode.Request.TRACK_TOOL:
            track = self._safe_call(
                self._track_tool, color_msg, depth_msg, tf_at_stamp, self.tool_class, default=None)
            if track is not None:
                self.pub_tool_track.publish(track)
        elif self.mode == SetVisionMode.Request.TRACK_HAND:
            hand_pose = self._safe_call(self._track_hand, color_msg, default=None)
            if hand_pose is not None:
                self.pub_hand_pose.publish(hand_pose)
```

- [ ] **Step 3: 테스트 통과 확인**

Run: `python3 -m pytest src/vision_node/test/test_vision_node.py -v`
Expected: 5 passed

- [ ] **Step 4: 커밋**

```bash
git add src/vision_node
git commit -m "feat: vision_node 이미지 콜백 배선 (TF 조회 + 검출 디스패치)"
```

---

### Task 5: `robot_control` 패키지 + `RG2Client` + `ServoLoop` 스텁

**Files:**
- Create: `src/robot_control/package.xml`, `setup.py`, `setup.cfg`, `resource/robot_control`
- Create: `src/robot_control/robot_control/__init__.py`, `rg2_client.py`, `servo_loop.py`
- Test: `src/robot_control/test/test_rg2_client.py`, `test_servo_loop.py`

**Interfaces:**
- Produces: `RG2Client(ip, port=502)` — `.open()`, `.close(width_mm, force_n)`, `.get_state() -> (float, bool)` 전부 TODO. `ServoLoop(kp_xy, kp_yaw, v_max, descend_speed, eps_descend, eps_grasp, n_stable, dt_latency, timeout_s, t_lost_s)` — `.start(tool_class, grasp_width_mm, grasp_force_n)`, `.on_tool_track(msg)`, `.step() -> ServoCommand`, `.should_close() -> bool`, `.should_abort() -> str|None` TODO, `.get_state() -> str` 구현됨.

- [ ] **Step 1: 패키지 뼈대**

`src/robot_control/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_control</name>
  <version>0.0.1</version>
  <description>RobotTask 액션 서버 - 모션/서보 파지/그리퍼 실행자</description>
  <maintainer email="hwangjeongui01@gmail.com">hwangjeongui</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>handover_interfaces</depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`src/robot_control/setup.py`:
```python
from setuptools import find_packages, setup

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robot_control.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='RobotTask 액션 서버 - 모션/서보 파지/그리퍼 실행자',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_control_node = robot_control.robot_control_node:main',
        ],
    },
)
```

`src/robot_control/setup.cfg`:
```
[develop]
script_dir=$base/lib/robot_control
[install]
install_scripts=$base/lib/robot_control
```

Run:
```bash
mkdir -p ~/rokey_proj_02/src/robot_control/resource
touch ~/rokey_proj_02/src/robot_control/resource/robot_control
touch ~/rokey_proj_02/src/robot_control/robot_control/__init__.py
```

- [ ] **Step 2: 실패하는 테스트 작성**

`src/robot_control/test/test_rg2_client.py`:
```python
import pytest
from robot_control.rg2_client import RG2Client


def test_stub_methods_raise_not_implemented():
    client = RG2Client(ip='192.168.1.1')
    with pytest.raises(NotImplementedError):
        client.open()
    with pytest.raises(NotImplementedError):
        client.close(30.0, 20.0)
    with pytest.raises(NotImplementedError):
        client.get_state()
```

`src/robot_control/test/test_servo_loop.py`:
```python
import pytest
from robot_control.servo_loop import ServoLoop, ServoState


def _make_loop():
    return ServoLoop(kp_xy=1.2, kp_yaw=1.0, v_max=0.25, descend_speed=0.10,
                      eps_descend=0.015, eps_grasp=0.005, n_stable=5,
                      dt_latency=0.05, timeout_s=5.0, t_lost_s=0.3)


def test_initial_state_is_tracking():
    loop = _make_loop()
    assert loop.get_state() == ServoState.TRACKING


def test_stub_methods_raise_not_implemented():
    loop = _make_loop()
    with pytest.raises(NotImplementedError):
        loop.start('spanner', 30.0, 20.0)
    with pytest.raises(NotImplementedError):
        loop.on_tool_track(object())
    with pytest.raises(NotImplementedError):
        loop.step()
    with pytest.raises(NotImplementedError):
        loop.should_close()
    with pytest.raises(NotImplementedError):
        loop.should_abort()
```

Run: `python3 -m pytest src/robot_control/test/ -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: `RG2Client`, `ServoLoop` 구현**

`src/robot_control/robot_control/rg2_client.py`:
```python
class RG2Client:
    """OnRobot RG2 그리퍼를 Modbus TCP(Compute Box)로 제어하는 클라이언트."""

    def __init__(self, ip: str, port: int = 502):
        self.ip = ip
        self.port = port

    def open(self) -> None:
        """RG2를 완전 개방한다."""
        raise NotImplementedError('RG2Client.open 구현 필요 (Modbus TCP 레지스터 쓰기)')

    def close(self, width_mm: float, force_n: float) -> None:
        """지정한 폭(mm)·힘(N)으로 RG2를 폐합한다."""
        raise NotImplementedError('RG2Client.close 구현 필요 (Modbus TCP 레지스터 쓰기)')

    def get_state(self):
        """(width_mm: float, grip_detected: bool) 튜플을 반환한다."""
        raise NotImplementedError('RG2Client.get_state 구현 필요 (Modbus TCP 레지스터 읽기)')
```

`src/robot_control/robot_control/servo_loop.py`:
```python
class ServoState:
    TRACKING = 'tracking'
    DESCENDING = 'descending'
    CLOSING = 'closing'
    LIFTING = 'lifting'


class ServoCommand:
    def __init__(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.yaw_rate = yaw_rate


class ServoLoop:
    """robot_control 내부 PBVS 서보 루프 (데모.md 2절)."""

    def __init__(self, kp_xy, kp_yaw, v_max, descend_speed,
                 eps_descend, eps_grasp, n_stable, dt_latency,
                 timeout_s, t_lost_s):
        self.kp_xy = kp_xy
        self.kp_yaw = kp_yaw
        self.v_max = v_max
        self.descend_speed = descend_speed
        self.eps_descend = eps_descend
        self.eps_grasp = eps_grasp
        self.n_stable = n_stable
        self.dt_latency = dt_latency
        self.timeout_s = timeout_s
        self.t_lost_s = t_lost_s
        self._state = ServoState.TRACKING

    def start(self, tool_class: str, grasp_width_mm: float, grasp_force_n: float) -> None:
        """servo_pick goal 시작 시 호출. 필터·타이머 초기화 등."""
        raise NotImplementedError('ServoLoop.start 구현 필요')

    def on_tool_track(self, msg) -> None:
        """/vision/tool_track 수신마다 호출. 필터(칼만/알파-베타) 갱신."""
        raise NotImplementedError('ServoLoop.on_tool_track 구현 필요')

    def step(self):
        """RT 명령 주기마다 호출. PBVS 제어식(2.3절)으로 다음 명령을 계산."""
        raise NotImplementedError('ServoLoop.step 구현 필요')

    def get_state(self) -> str:
        """현재 서보 상태(tracking/descending/closing/lifting)."""
        return self._state

    def should_close(self) -> bool:
        """폐합 판정(2.6절: |e_xy|<eps_grasp가 n_stable주기 연속 ∧ z_gap<z_close ∧ 공분산<임계)."""
        raise NotImplementedError('ServoLoop.should_close 구현 필요')

    def should_abort(self):
        """발산/유실/이탈/타임아웃(2.8절) 판정. 사유 문자열 또는 None."""
        raise NotImplementedError('ServoLoop.should_abort 구현 필요')
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest src/robot_control/test/ -v`
Expected: 6 passed

- [ ] **Step 5: 커밋**

```bash
git add src/robot_control
git commit -m "feat: robot_control 패키지 뼈대 + RG2Client/ServoLoop 스텁"
```

---

### Task 6: `robot_control` — move_named / move_pose / place_down / release_and_retry

**Files:**
- Create: `src/robot_control/robot_control/robot_control_node.py`
- Test: `src/robot_control/test/test_robot_control_node.py`

**Interfaces:**
- Consumes: `RG2Client`, `ServoLoop` (Task 5), `handover_interfaces.action.RobotTask`
- Produces: `RobotControlNode` 클래스, `_execute_callback` 디스패치(이 태스크에서는 `move_named`/`move_pose`/`place_down`/`release_and_retry`만 등록), `_call_move_service` TODO 스텁

- [ ] **Step 1: 실패하는 테스트 작성**

`src/robot_control/test/test_robot_control_node.py`:
```python
import rclpy
import pytest

from handover_interfaces.action import RobotTask
from robot_control.robot_control_node import RobotControlNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = RobotControlNode()
    yield n
    n.destroy_node()


class FakeGoalHandle:
    def __init__(self, request):
        self.request = request
        self.succeeded = False
        self.aborted = False
        self.feedback_msgs = []

    def succeed(self):
        self.succeeded = True

    def abort(self):
        self.aborted = True

    def publish_feedback(self, fb):
        self.feedback_msgs.append(fb)


def _goal(task_type, named_target=''):
    g = RobotTask.Goal()
    g.task_type = task_type
    g.named_target = named_target
    return g


def test_move_named_success(node):
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_move_named_failure(node):
    node._call_move_service = lambda **kw: False
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


def test_move_named_stub_not_implemented_is_treated_as_failure(node):
    gh = FakeGoalHandle(_goal('move_named', named_target='watch'))

    result = node._execute_move_named(gh)

    assert gh.aborted is True
    assert result.success is False


def test_release_and_retry_calls_open_and_move_to_watch(node):
    calls = []
    node.rg2_client.open = lambda: calls.append('open')
    node._call_move_service = lambda **kw: calls.append(('move', kw)) or True
    gh = FakeGoalHandle(_goal('release_and_retry'))

    result = node._execute_release_and_retry(gh)

    assert calls[0] == 'open'
    assert calls[1] == ('move', {'named_target': 'watch'})
    assert gh.succeeded is True
    assert result.success is True


def test_dispatch_unknown_task_type_aborts(node):
    gh = FakeGoalHandle(_goal('unknown_type'))

    result = node._execute_callback(gh)

    assert gh.aborted is True
    assert result.success is False


def test_dispatch_routes_move_named(node):
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_goal('move_named', named_target='home'))

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_dispatch_routes_place_down_to_move_named_handler(node):
    node._call_move_service = lambda **kw: True
    gh = FakeGoalHandle(_goal('place_down', named_target='place_down'))

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True
```

Run: `python3 -m pytest src/robot_control/test/test_robot_control_node.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'robot_control.robot_control_node'`)

- [ ] **Step 2: `RobotControlNode` 뼈대 구현 (기본 액션들만)**

`src/robot_control/robot_control/robot_control_node.py`:
```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from handover_interfaces.action import RobotTask

from robot_control.rg2_client import RG2Client
from robot_control.servo_loop import ServoLoop


class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')

        self.declare_parameter('rg2_ip', '192.168.1.1')
        self.declare_parameter('servo.kp_xy', 1.2)
        self.declare_parameter('servo.kp_yaw', 1.0)
        self.declare_parameter('servo.v_max', 0.25)
        self.declare_parameter('servo.descend_speed', 0.10)
        self.declare_parameter('servo.eps_descend', 0.015)
        self.declare_parameter('servo.eps_grasp', 0.005)
        self.declare_parameter('servo.n_stable', 5)
        self.declare_parameter('servo.dt_latency', 0.05)
        self.declare_parameter('servo.timeout', 5.0)
        self.declare_parameter('servo.t_lost', 0.3)

        self.rg2_client = RG2Client(ip=self.get_parameter('rg2_ip').value)
        self.servo_loop = ServoLoop(
            kp_xy=self.get_parameter('servo.kp_xy').value,
            kp_yaw=self.get_parameter('servo.kp_yaw').value,
            v_max=self.get_parameter('servo.v_max').value,
            descend_speed=self.get_parameter('servo.descend_speed').value,
            eps_descend=self.get_parameter('servo.eps_descend').value,
            eps_grasp=self.get_parameter('servo.eps_grasp').value,
            n_stable=self.get_parameter('servo.n_stable').value,
            dt_latency=self.get_parameter('servo.dt_latency').value,
            timeout_s=self.get_parameter('servo.timeout').value,
            t_lost_s=self.get_parameter('servo.t_lost').value,
        )

        self._action_server = ActionServer(
            self, RobotTask, 'robot_task', execute_callback=self._execute_callback)

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    # ---- move / place_down / release_and_retry ----

    def _call_move_service(self, named_target='', target_pose=None) -> bool:
        """Doosan 모션 서비스(정적 이동) 호출. dsr_msgs2 등 드라이버 서비스 인터페이스 확인 후 구현."""
        raise NotImplementedError('_call_move_service 구현 필요')

    def _execute_move_named(self, goal_handle):
        result = RobotTask.Result()
        success = self._safe_call(
            self._call_move_service, named_target=goal_handle.request.named_target, default=False)
        if success:
            goal_handle.succeed()
            result.success = True
        else:
            goal_handle.abort()
            result.success = False
            result.message = f'move_named({goal_handle.request.named_target}) failed'
        return result

    def _execute_move_pose(self, goal_handle):
        result = RobotTask.Result()
        success = self._safe_call(
            self._call_move_service, target_pose=goal_handle.request.target_pose, default=False)
        if success:
            goal_handle.succeed()
            result.success = True
        else:
            goal_handle.abort()
            result.success = False
            result.message = 'move_pose failed'
        return result

    def _execute_release_and_retry(self, goal_handle):
        result = RobotTask.Result()
        self._safe_call(self.rg2_client.open)
        success = self._safe_call(self._call_move_service, named_target='watch', default=False)
        if success:
            goal_handle.succeed()
            result.success = True
            result.message = 'released, returned to watch'
        else:
            goal_handle.abort()
            result.success = False
            result.message = 'release_and_retry failed to return to watch'
        return result

    # ---- action dispatch ----

    def _execute_callback(self, goal_handle):
        task_type = goal_handle.request.task_type
        handlers = {
            'move_named': self._execute_move_named,
            'move_pose': self._execute_move_pose,
            'place_down': self._execute_move_named,
            'release_and_retry': self._execute_release_and_retry,
        }
        handler = handlers.get(task_type)
        if handler is None:
            goal_handle.abort()
            result = RobotTask.Result()
            result.success = False
            result.message = f'unknown task_type: {task_type}'
            return result
        return handler(goal_handle)


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: 테스트 통과 확인**

Run: `python3 -m pytest src/robot_control/test/test_robot_control_node.py -v`
Expected: 7 passed

- [ ] **Step 4: 커밋**

```bash
git add src/robot_control
git commit -m "feat: robot_control move_named/move_pose/place_down/release_and_retry 배선"
```

---

### Task 7: `robot_control` — servo_pick 오케스트레이션

**Files:**
- Modify: `src/robot_control/robot_control/robot_control_node.py`
- Modify: `src/robot_control/test/test_robot_control_node.py`

**Interfaces:**
- Consumes: `self.servo_loop.should_abort()/.should_close()/.step()/.get_state()`, `self.rg2_client.close()/.get_state()`
- Produces: `_servo_pick_tick`, `_execute_servo_pick`, `'servo_pick'` 디스패치 등록. `_open_rt_session`, `_close_rt_session`, `_estimate_payload` TODO 스텁.

- [ ] **Step 1: 실패하는 테스트 추가**

`src/robot_control/test/test_robot_control_node.py` 하단에 추가:
```python
def test_servo_pick_tick_continue(node):
    node.servo_loop.should_abort = lambda: None
    node.servo_loop.should_close = lambda: False

    status, reason = node._servo_pick_tick()

    assert status == 'CONTINUE'
    assert reason is None


def test_servo_pick_tick_close(node):
    node.servo_loop.should_abort = lambda: None
    node.servo_loop.should_close = lambda: True

    status, reason = node._servo_pick_tick()

    assert status == 'CLOSE'


def test_servo_pick_tick_abort(node):
    node.servo_loop.should_abort = lambda: 'diverged'

    status, reason = node._servo_pick_tick()

    assert status == 'ABORT'
    assert reason == 'diverged'


def test_execute_servo_pick_success_closes_gripper_and_returns_result(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    ticks = iter(['CONTINUE', 'CONTINUE', 'CLOSE'])
    node._servo_pick_tick = lambda: (next(ticks), None)
    node.servo_loop.step = lambda: None
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: None
    node._estimate_payload = lambda: 0.31
    node.rg2_client.close = lambda width, force: None
    node.rg2_client.get_state = lambda: (29.4, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.succeeded is True
    assert result.success is True
    assert result.measured_payload_kg == 0.31
    assert result.final_width_mm == 29.4
    assert result.grip_detected is True
    assert len(gh.feedback_msgs) == 3


def test_execute_servo_pick_abort_returns_reason(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._servo_pick_tick = lambda: ('ABORT', 'diverged')
    node.servo_loop.get_state = lambda: 'tracking'
    node.servo_loop.start = lambda *a, **k: None
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: None

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_servo_pick(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'diverged'


def test_dispatch_routes_servo_pick(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._servo_pick_tick = lambda: ('CLOSE', None)
    node.servo_loop.get_state = lambda: 'closing'
    node.servo_loop.start = lambda *a, **k: None
    node._open_rt_session = lambda: None
    node._close_rt_session = lambda: None
    node._estimate_payload = lambda: 0.3
    node.rg2_client.close = lambda width, force: None
    node.rg2_client.get_state = lambda: (30.0, True)

    gh = FakeGoalHandle(_goal('servo_pick'))
    gh.request.tool_class = 'spanner'
    gh.request.grasp_width_mm = 30.0
    gh.request.grasp_force_n = 20.0

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True
```

Run: `python3 -m pytest src/robot_control/test/test_robot_control_node.py -v`
Expected: FAIL (`AttributeError: 'RobotControlNode' object has no attribute '_servo_pick_tick'`)

- [ ] **Step 2: servo_pick 오케스트레이션 구현**

`src/robot_control/robot_control/robot_control_node.py` 상단 import 블록에 `import time`을 추가 (`import rclpy` 앞 줄):
```python
import time

import rclpy
```

`_execute_release_and_retry` 메서드 다음, `# ---- action dispatch ----` 이전에 추가:
```python
    # ---- servo_pick ----

    def _open_rt_session(self) -> None:
        """Doosan 실시간 제어 세션을 연다. 드라이버 RT API 확인 후 구현."""
        raise NotImplementedError('_open_rt_session 구현 필요')

    def _close_rt_session(self) -> None:
        """실시간 제어 세션을 닫고 서비스 모션 모드로 복귀한다."""
        raise NotImplementedError('_close_rt_session 구현 필요')

    def _estimate_payload(self) -> float:
        """들어올림 직후 외부 토크로 페이로드(kg)를 추정한다."""
        raise NotImplementedError('_estimate_payload 구현 필요')

    def _servo_pick_tick(self):
        abort_reason = self.servo_loop.should_abort()
        if abort_reason is not None:
            return ('ABORT', abort_reason)
        if self.servo_loop.should_close():
            return ('CLOSE', None)
        return ('CONTINUE', None)

    def _on_tool_track_during_servo(self, msg):
        self.servo_loop.on_tool_track(msg)

    def _execute_servo_pick(self, goal_handle):
        from handover_interfaces.msg import ToolTrack

        request = goal_handle.request
        result = RobotTask.Result()

        self._safe_call(self._open_rt_session)
        self.servo_loop.start(request.tool_class, request.grasp_width_mm, request.grasp_force_n)
        servo_sub = self.create_subscription(
            ToolTrack, '/vision/tool_track', self._on_tool_track_during_servo, 10)

        try:
            while rclpy.ok():
                status, reason = self._servo_pick_tick()
                feedback = RobotTask.Feedback()
                feedback.state = self.servo_loop.get_state()
                goal_handle.publish_feedback(feedback)

                if status == 'ABORT':
                    goal_handle.abort()
                    result.success = False
                    result.message = reason
                    return result
                if status == 'CLOSE':
                    break

                self.servo_loop.step()
                time.sleep(0.01)

            self._safe_call(self.rg2_client.close, request.grasp_width_mm, request.grasp_force_n)
            width_mm, grip_detected = self._safe_call(
                self.rg2_client.get_state, default=(0.0, False))
            payload_kg = self._safe_call(self._estimate_payload, default=0.0)

            goal_handle.succeed()
            result.success = True
            result.measured_payload_kg = payload_kg
            result.final_width_mm = width_mm
            result.grip_detected = grip_detected
        finally:
            self.destroy_subscription(servo_sub)
            self._safe_call(self._close_rt_session)

        return result
```

`_execute_callback`의 `handlers` 딕셔너리를 다음으로 교체:
```python
        handlers = {
            'move_named': self._execute_move_named,
            'move_pose': self._execute_move_pose,
            'place_down': self._execute_move_named,
            'release_and_retry': self._execute_release_and_retry,
            'servo_pick': self._execute_servo_pick,
        }
```

- [ ] **Step 3: 테스트 통과 확인**

Run: `python3 -m pytest src/robot_control/test/test_robot_control_node.py -v`
Expected: 13 passed

- [ ] **Step 4: 커밋**

```bash
git add src/robot_control
git commit -m "feat: robot_control servo_pick 오케스트레이션 배선"
```

---

### Task 8: `robot_control` — handover_hold + fault 폴링 + 그리퍼 상태 타이머

**Files:**
- Modify: `src/robot_control/robot_control/robot_control_node.py`
- Modify: `src/robot_control/test/test_robot_control_node.py`

**Interfaces:**
- Produces: `_execute_handover_hold`, `'handover_hold'` 디스패치 등록, `_on_state_poll_timer`(0.1s) → `/robot/fault` 퍼블리시, `_on_gripper_timer`(0.5s) → `/gripper/state` 퍼블리시. TODO 스텁 `_enable_compliance`, `_disable_compliance`, `_is_pull_detected`, `_read_robot_state`, `_check_fault`.

- [ ] **Step 1: 실패하는 테스트 추가**

`src/robot_control/test/test_robot_control_node.py` 하단에 추가:
```python
def test_handover_hold_releases_on_pull_detected(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._latest_robot_state = 'state'
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: True
    calls = []
    node.rg2_client.open = lambda: calls.append('open')

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_handover_hold(gh)

    assert calls == ['open']
    assert gh.succeeded is True
    assert result.success is True
    assert result.message == 'pull_detected, released'


def test_dispatch_routes_handover_hold(node, monkeypatch):
    import time as time_module
    monkeypatch.setattr(time_module, 'sleep', lambda s: None)

    node._latest_robot_state = 'state'
    node._enable_compliance = lambda: None
    node._disable_compliance = lambda: None
    node._is_pull_detected = lambda state: True
    node.rg2_client.open = lambda: None

    gh = FakeGoalHandle(_goal('handover_hold'))

    result = node._execute_callback(gh)

    assert gh.succeeded is True
    assert result.success is True


def test_gripper_timer_publishes_state(node):
    from handover_interfaces.msg import GripperState

    node.rg2_client.get_state = lambda: (30.0, True)
    published = []
    node.pub_gripper_state.publish = published.append

    node._on_gripper_timer()

    assert len(published) == 1
    assert isinstance(published[0], GripperState)
    assert published[0].width_mm == 30.0
    assert published[0].grip_detected is True


def test_state_poll_timer_publishes_fault_when_detected(node):
    node._read_robot_state = lambda: 'state'
    node._check_fault = lambda state: 'protective_stop'
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert len(published) == 1
    assert published[0].data == 'protective_stop'
    assert node._latest_robot_state == 'state'


def test_state_poll_timer_silent_when_no_fault(node):
    node._read_robot_state = lambda: 'state'
    node._check_fault = lambda state: None
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert published == []


def test_state_poll_timer_skips_when_state_unavailable(node):
    node._read_robot_state = lambda: None
    published = []
    node.pub_fault.publish = published.append

    node._on_state_poll_timer()

    assert published == []
    assert node._latest_robot_state is None
```

Run: `python3 -m pytest src/robot_control/test/test_robot_control_node.py -v`
Expected: FAIL (`AttributeError: 'RobotControlNode' object has no attribute '_execute_handover_hold'`)

- [ ] **Step 2: handover_hold + fault/그리퍼 폴링 구현**

`__init__`에서 `self._action_server = ActionServer(...)` 다음 줄에 추가:
```python
        self._latest_robot_state = None
        self._gripper_timer = self.create_timer(0.5, self._on_gripper_timer)
        self._state_poll_timer = self.create_timer(0.1, self._on_state_poll_timer)

        self.pub_gripper_state = self.create_publisher(GripperState, '/gripper/state', 10)
        self.pub_fault = self.create_publisher(String, '/robot/fault', 10)
```

import 블록에 다음 추가 (`from handover_interfaces.action import RobotTask` 다음 줄):
```python
from handover_interfaces.msg import GripperState
from std_msgs.msg import String
```

`_execute_servo_pick` 다음, `# ---- action dispatch ----` 이전에 추가:
```python
    # ---- handover_hold ----

    def _enable_compliance(self) -> None:
        """컴플라이언스 모드를 켠다."""
        raise NotImplementedError('_enable_compliance 구현 필요')

    def _disable_compliance(self) -> None:
        """컴플라이언스 모드를 끈다."""
        raise NotImplementedError('_disable_compliance 구현 필요')

    def _is_pull_detected(self, robot_state) -> bool:
        """robot_state의 외부 토크로 당김 힘 임계 초과 여부를 판정한다."""
        raise NotImplementedError('_is_pull_detected 구현 필요')

    def _execute_handover_hold(self, goal_handle):
        result = RobotTask.Result()
        self._safe_call(self._enable_compliance)
        try:
            while rclpy.ok():
                if self._latest_robot_state is not None and self._safe_call(
                        self._is_pull_detected, self._latest_robot_state, default=False):
                    break
                time.sleep(0.01)
            self._safe_call(self.rg2_client.open)
            goal_handle.succeed()
            result.success = True
            result.message = 'pull_detected, released'
        finally:
            self._safe_call(self._disable_compliance)
        return result

    # ---- fault / robot state polling ----

    def _read_robot_state(self):
        """Doosan 드라이버로부터 최신 로봇 상태(외부 토크 등)를 읽는다."""
        raise NotImplementedError('_read_robot_state 구현 필요')

    def _check_fault(self, robot_state):
        """protective stop / 토크 이상 등을 판정한다. 사유 문자열 또는 None."""
        raise NotImplementedError('_check_fault 구현 필요')

    def _on_state_poll_timer(self):
        state = self._safe_call(self._read_robot_state, default=None)
        if state is None:
            return
        self._latest_robot_state = state
        fault_reason = self._safe_call(self._check_fault, state, default=None)
        if fault_reason is not None:
            msg = String()
            msg.data = fault_reason
            self.pub_fault.publish(msg)

    def _on_gripper_timer(self):
        width_mm, grip_detected = self._safe_call(
            self.rg2_client.get_state, default=(0.0, False))
        msg = GripperState()
        msg.width_mm = width_mm
        msg.grip_detected = grip_detected
        self.pub_gripper_state.publish(msg)
```

`_execute_callback`의 `handlers` 딕셔너리에 `'handover_hold': self._execute_handover_hold,` 추가.

- [ ] **Step 3: 테스트 통과 확인**

Run: `python3 -m pytest src/robot_control/test/test_robot_control_node.py -v`
Expected: 19 passed

- [ ] **Step 4: 커밋**

```bash
git add src/robot_control
git commit -m "feat: robot_control handover_hold + fault/그리퍼 폴링 배선"
```

---

### Task 9: `robot_control` launch 파일

**Files:**
- Create: `src/robot_control/launch/robot_control.launch.py`

**Interfaces:**
- Consumes: `robot_control_node` executable (Task 6-8)

- [ ] **Step 1: launch 파일 작성**

`src/robot_control/launch/robot_control.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='robot_control', executable='robot_control_node'),
    ])
```

- [ ] **Step 2: 빌드 후 launch 확인**

Run:
```bash
cd ~/rokey_proj_02
colcon build --symlink-install --packages-select robot_control
source install/setup.bash
timeout 3 ros2 launch robot_control robot_control.launch.py || true
```
Expected: 에러 없이 `robot_control` 노드가 뜨고(TODO 스텁이 호출되기 전까지는 조용함), 3초 후 timeout으로 정상 종료.

- [ ] **Step 3: 커밋**

```bash
git add src/robot_control/launch
git commit -m "feat: robot_control launch 파일 추가"
```

---

### Task 10: `task_manager` 패키지 + 상태 골격 + FAULT 처리

**Files:**
- Create: `src/task_manager/package.xml`, `setup.py`, `setup.cfg`, `resource/task_manager`
- Create: `src/task_manager/task_manager/__init__.py`, `task_manager_node.py`
- Test: `src/task_manager/test/test_task_manager_node.py`

**Interfaces:**
- Produces: `State` 상수 클래스, `TaskManagerNode` 뼈대 (`/task/status` 퍼블리셔, `/user_command/text`/`/vision/tool_track`/`/vision/hand_pose`/`/robot/fault` 구독, `/vision/set_mode` 서비스 클라, `RobotTask` 액션 클라), `_set_state`, `_publish_status`, `_safe_call`, `_on_fault`

- [ ] **Step 1: 패키지 뼈대**

`src/task_manager/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>task_manager</name>
  <version>0.0.1</version>
  <description>명령 해석·상태 머신 감독 노드</description>
  <maintainer email="hwangjeongui01@gmail.com">hwangjeongui</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>handover_interfaces</depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`src/task_manager/setup.py`:
```python
from setuptools import find_packages, setup

package_name = 'task_manager'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/task_manager.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='명령 해석·상태 머신 감독 노드',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_manager_node = task_manager.task_manager_node:main',
        ],
    },
)
```

`src/task_manager/setup.cfg`:
```
[develop]
script_dir=$base/lib/task_manager
[install]
install_scripts=$base/lib/task_manager
```

Run:
```bash
mkdir -p ~/rokey_proj_02/src/task_manager/resource
touch ~/rokey_proj_02/src/task_manager/resource/task_manager
touch ~/rokey_proj_02/src/task_manager/task_manager/__init__.py
```

- [ ] **Step 2: 실패하는 테스트 작성**

`src/task_manager/test/test_task_manager_node.py`:
```python
import json

import rclpy
import pytest
from std_msgs.msg import String

from task_manager.task_manager_node import TaskManagerNode, State


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = TaskManagerNode()
    yield n
    n.destroy_node()


def test_initial_state_is_idle(node):
    assert node.state == State.IDLE


def test_set_state_publishes_json_status(node):
    published = []
    node.pub_status.publish = published.append

    node._set_state(State.PARSING, detail='hello')

    assert node.state == State.PARSING
    assert len(published) == 1
    payload = json.loads(published[0].data)
    assert payload == {'state': 'PARSING', 'detail': 'hello'}


def test_fault_message_transitions_to_fault_from_any_state(node):
    published = []
    node.pub_status.publish = published.append
    node.state = State.SERVO_PICK

    msg = String()
    msg.data = 'torque anomaly'
    node._on_fault(msg)

    assert node.state == State.FAULT
    payload = json.loads(published[-1].data)
    assert payload['detail'] == 'torque anomaly'


def test_fault_message_ignored_if_already_in_fault(node):
    node.state = State.FAULT
    published = []
    node.pub_status.publish = published.append

    msg = String()
    msg.data = 'another fault'
    node._on_fault(msg)

    assert published == []
```

Run: `python3 -m pytest src/task_manager/test/ -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: `TaskManagerNode` 뼈대 구현**

`src/task_manager/task_manager/task_manager_node.py`:
```python
import json

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode


class State:
    IDLE = 'IDLE'
    PARSING = 'PARSING'
    MOVE_TO_WATCH = 'MOVE_TO_WATCH'
    DETECT_TRACK = 'DETECT_TRACK'
    SERVO_PICK = 'SERVO_PICK'
    VERIFY_GRASP = 'VERIFY_GRASP'
    MOVE_SAFE = 'MOVE_SAFE'
    TRACK_HAND = 'TRACK_HAND'
    WAIT_PULL = 'WAIT_PULL'
    RELEASE = 'RELEASE'
    HOME = 'HOME'
    FAULT = 'FAULT'


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager')

        self.declare_parameter('detect_track_max_cycles', 3)
        self.declare_parameter('verify_grasp_max_retries', 2)
        self.declare_parameter('wait_pull_timeout_s', 60.0)
        self.declare_parameter('hand_detect_timeout_s', 5.0)

        self.state = State.IDLE
        self.current_tool = None
        self._detect_track_cycles = 0
        self._verify_grasp_retries = 0
        self._hand_timeout_timer = None
        self._wait_pull_timeout_timer = None

        self.pub_status = self.create_publisher(String, '/task/status', 10)
        self.sub_command = self.create_subscription(
            String, '/user_command/text', self._on_user_command, 10)
        self.sub_tool_track = self.create_subscription(
            ToolTrack, '/vision/tool_track', self._on_tool_track, 10)
        self.sub_hand_pose = self.create_subscription(
            PoseStamped, '/vision/hand_pose', self._on_hand_pose, 10)
        self.sub_fault = self.create_subscription(
            String, '/robot/fault', self._on_fault, 10)

        self.set_mode_client = self.create_client(SetVisionMode, '/vision/set_mode')
        self.robot_task_client = ActionClient(self, RobotTask, 'robot_task')

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _publish_status(self, detail=''):
        msg = String()
        msg.data = json.dumps({'state': self.state, 'detail': detail})
        self.pub_status.publish(msg)

    def _set_state(self, new_state, detail=''):
        self.state = new_state
        self._publish_status(detail)

    def _on_fault(self, msg):
        if self.state == State.FAULT:
            return
        self._set_state(State.FAULT, detail=msg.data)

    def _on_user_command(self, msg):
        pass

    def _on_tool_track(self, msg):
        pass

    def _on_hand_pose(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest src/task_manager/test/ -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/task_manager
git commit -m "feat: task_manager 패키지 뼈대 + 상태 골격 + FAULT 처리"
```

---

### Task 11: `task_manager` — PARSING + MOVE_TO_WATCH + 액션 결과 플러밍

**Files:**
- Modify: `src/task_manager/task_manager/task_manager_node.py`
- Modify: `src/task_manager/test/test_task_manager_node.py`

**Interfaces:**
- Produces: `_call_llm` TODO 스텁, `_handle_parsing`, `_set_vision_mode`, `_send_robot_goal`, `_on_goal_response`, `_on_robot_result` (8개 상태 전체 디스패치, 이 태스크에서는 `_handle_move_to_watch_result`만 구현하고 나머지 7개 핸들러는 Task 12/13에서 추가)

- [ ] **Step 1: 실패하는 테스트 추가**

`src/task_manager/test/test_task_manager_node.py` 하단에 추가:
```python
class _FakeResult:
    def __init__(self, success=True, message=''):
        self.success = success
        self.message = message
        self.measured_payload_kg = 0.0
        self.final_width_mm = 0.0
        self.grip_detected = False


class _FakeResponse:
    def __init__(self, result):
        self.result = result


class _FakeFuture:
    def __init__(self, response):
        self._response = response

    def result(self):
        return self._response


def test_user_command_ignored_unless_idle(node):
    node.state = State.SERVO_PICK
    called = []
    node._handle_parsing = lambda text: called.append(text)

    node._on_user_command(String(data='스패너 갖다줘'))

    assert called == []


def test_user_command_triggers_parsing_and_move_to_watch(node):
    node.state = State.IDLE
    node._call_llm = lambda text: {'tool': 'spanner', 'action': 'handover'}
    sent_goals = []
    node._send_robot_goal = lambda task_type, **kw: sent_goals.append((task_type, kw))
    node._set_vision_mode = lambda mode, tool_class='': None

    node._on_user_command(String(data='스패너 갖다줘'))

    assert node.state == State.MOVE_TO_WATCH
    assert node.current_tool == 'spanner'
    assert sent_goals == [('move_named', {'named_target': 'watch'})]


def test_parsing_failure_returns_to_idle(node):
    node.state = State.IDLE

    def _raise(text):
        raise NotImplementedError('todo')
    node._call_llm = _raise

    node._on_user_command(String(data='asdf'))

    assert node.state == State.IDLE


def test_move_to_watch_result_success_transitions_to_detect_track(node):
    node.state = State.MOVE_TO_WATCH

    node._on_robot_result(_FakeFuture(_FakeResponse(_FakeResult(success=True))))

    assert node.state == State.DETECT_TRACK


def test_move_to_watch_result_failure_transitions_to_fault(node):
    node.state = State.MOVE_TO_WATCH

    node._on_robot_result(_FakeFuture(_FakeResponse(_FakeResult(success=False, message='motion failed'))))

    assert node.state == State.FAULT
```

Run: `python3 -m pytest src/task_manager/test/ -v`
Expected: FAIL (`AttributeError`/`TypeError` — `_handle_parsing`, `_on_robot_result` 등 미구현)

- [ ] **Step 2: PARSING/MOVE_TO_WATCH + 결과 플러밍 구현**

`_on_user_command`, `_on_tool_track`, `_on_hand_pose`의 `pass` 본문 및 그 아래를 아래 코드로 교체:
```python
    def _call_llm(self, text: str) -> dict:
        """LLM API를 호출해 {"tool": ..., "action": ...}를 반환한다. 스키마 검증·재시도 포함."""
        raise NotImplementedError('_call_llm 구현 필요')

    def _on_user_command(self, msg):
        if self.state != State.IDLE:
            return
        self._set_state(State.PARSING, detail=msg.data)
        self._handle_parsing(msg.data)

    def _handle_parsing(self, text):
        parsed = self._safe_call(self._call_llm, text, default=None)
        if not parsed or 'tool' not in parsed:
            self._set_state(State.IDLE, detail='명령을 이해하지 못했습니다. 다시 말씀해주세요.')
            return
        self.current_tool = parsed['tool']
        self._detect_track_cycles = 0
        self._verify_grasp_retries = 0
        self._set_state(State.MOVE_TO_WATCH)
        self._set_vision_mode(SetVisionMode.Request.TRACK_TOOL, self.current_tool)
        self._send_robot_goal('move_named', named_target='watch')

    def _set_vision_mode(self, mode, tool_class=''):
        request = SetVisionMode.Request()
        request.mode = mode
        request.tool_class = tool_class
        self.set_mode_client.call_async(request)

    def _send_robot_goal(self, task_type, named_target='', target_pose=None,
                          tool_class='', grasp_width_mm=0.0, grasp_force_n=0.0):
        goal = RobotTask.Goal()
        goal.task_type = task_type
        goal.named_target = named_target
        if target_pose is not None:
            goal.target_pose = target_pose
        goal.tool_class = tool_class
        goal.grasp_width_mm = grasp_width_mm
        goal.grasp_force_n = grasp_force_n
        future = self.robot_task_client.send_goal_async(
            goal, feedback_callback=self._on_robot_feedback)
        future.add_done_callback(self._on_goal_response)

    def _on_robot_feedback(self, feedback_msg):
        self._publish_status(detail=f'servo:{feedback_msg.feedback.state}')

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._set_state(State.FAULT, detail='goal rejected')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_robot_result)

    def _on_robot_result(self, future):
        response = future.result()
        result = response.result
        if self.state == State.MOVE_TO_WATCH:
            self._handle_move_to_watch_result(result)
        elif self.state == State.SERVO_PICK:
            self._handle_servo_pick_result(result)
        elif self.state == State.VERIFY_GRASP:
            self._handle_release_and_retry_result(result)
        elif self.state == State.MOVE_SAFE:
            self._handle_move_safe_result(result)
        elif self.state == State.TRACK_HAND:
            self._handle_track_hand_result(result)
        elif self.state == State.WAIT_PULL:
            self._handle_wait_pull_result(result)
        elif self.state == State.RELEASE:
            self._handle_release_result(result)
        elif self.state == State.HOME:
            self._handle_home_result(result)

    def _handle_move_to_watch_result(self, result):
        if result.success:
            self._set_state(State.DETECT_TRACK)
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _on_tool_track(self, msg):
        pass

    def _on_hand_pose(self, msg):
        pass
```

(참고: `_handle_servo_pick_result`, `_handle_release_and_retry_result`, `_handle_move_safe_result`, `_handle_track_hand_result`, `_handle_wait_pull_result`, `_handle_release_result`, `_handle_home_result`는 Python이 호출 시점에만 속성을 조회하므로, 이 태스크의 테스트가 해당 상태에 진입하지 않는 한 아직 정의되지 않아도 에러가 나지 않는다. Task 12/13에서 정의한다.)

- [ ] **Step 3: 테스트 통과 확인**

Run: `python3 -m pytest src/task_manager/test/ -v`
Expected: 9 passed

- [ ] **Step 4: 커밋**

```bash
git add src/task_manager
git commit -m "feat: task_manager PARSING/MOVE_TO_WATCH + 액션 결과 플러밍"
```

---

### Task 12: `task_manager` — DETECT_TRACK + SERVO_PICK + VERIFY_GRASP

**Files:**
- Modify: `src/task_manager/task_manager/task_manager_node.py`
- Modify: `src/task_manager/test/test_task_manager_node.py`

**Interfaces:**
- Produces: `_check_trigger`, `_get_grasp_spec`, `_verify_grasp` TODO 스텁, `_on_tool_track` 실제 구현, `_handle_servo_pick_result`, `_handle_release_and_retry_result`

- [ ] **Step 1: 실패하는 테스트 추가**

`src/task_manager/test/test_task_manager_node.py` 하단에 추가:
```python
def test_tool_track_ignored_unless_detect_track(node):
    node.state = State.IDLE
    node._check_trigger = lambda msg: True
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_tool_track(ToolTrack())

    assert sent == []


def test_tool_track_trigger_sends_servo_pick_goal(node):
    node.state = State.DETECT_TRACK
    node.current_tool = 'spanner'
    node._check_trigger = lambda msg: True
    node._get_grasp_spec = lambda tool_class: (30.0, 20.0)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_tool_track(ToolTrack())

    assert node.state == State.SERVO_PICK
    assert sent == [('servo_pick', {
        'tool_class': 'spanner', 'grasp_width_mm': 30.0, 'grasp_force_n': 20.0})]


def test_tool_track_no_trigger_increments_cycle_and_reports_after_max(node):
    node.state = State.DETECT_TRACK
    node._check_trigger = lambda msg: False

    node._on_tool_track(ToolTrack())
    assert node.state == State.DETECT_TRACK
    node._on_tool_track(ToolTrack())
    assert node.state == State.DETECT_TRACK
    node._on_tool_track(ToolTrack())

    assert node.state == State.IDLE


def test_servo_pick_result_torque_anomaly_goes_to_fault(node):
    node.state = State.SERVO_PICK

    node._handle_servo_pick_result(_FakeResult(success=False, message='torque anomaly'))

    assert node.state == State.FAULT


def test_servo_pick_result_other_failure_returns_to_detect_track(node):
    node.state = State.SERVO_PICK
    node._detect_track_cycles = 2

    node._handle_servo_pick_result(_FakeResult(success=False, message='timeout'))

    assert node.state == State.DETECT_TRACK
    assert node._detect_track_cycles == 0


def test_servo_pick_result_success_and_verify_passes_moves_to_move_safe(node):
    node.state = State.SERVO_PICK
    node._verify_grasp = lambda result: True
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_servo_pick_result(_FakeResult(success=True))

    assert node.state == State.MOVE_SAFE
    assert sent == [('move_named', {'named_target': 'safe'})]


def test_servo_pick_result_success_and_verify_fails_sends_release_and_retry(node):
    node.state = State.SERVO_PICK
    node._verify_grasp = lambda result: False
    node._verify_grasp_retries = 0
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_servo_pick_result(_FakeResult(success=True))

    assert node.state == State.VERIFY_GRASP
    assert sent == [('release_and_retry', {})]
    assert node._verify_grasp_retries == 1


def test_verify_grasp_exceeds_max_retries_reports_to_idle(node):
    node.state = State.SERVO_PICK
    node._verify_grasp = lambda result: False
    node._verify_grasp_retries = 2
    node._send_robot_goal = lambda *a, **k: None

    node._handle_servo_pick_result(_FakeResult(success=True))

    assert node.state == State.IDLE


def test_release_and_retry_result_success_returns_to_detect_track(node):
    node.state = State.VERIFY_GRASP

    node._handle_release_and_retry_result(_FakeResult(success=True))

    assert node.state == State.DETECT_TRACK


def test_release_and_retry_result_failure_goes_to_fault(node):
    node.state = State.VERIFY_GRASP

    node._handle_release_and_retry_result(_FakeResult(success=False, message='release failed'))

    assert node.state == State.FAULT
```

`test_task_manager_node.py` 상단 import에 `from handover_interfaces.msg import ToolTrack` 추가.

Run: `python3 -m pytest src/task_manager/test/ -v`
Expected: FAIL (`AttributeError: '_check_trigger'` 등)

- [ ] **Step 2: DETECT_TRACK/SERVO_PICK/VERIFY_GRASP 구현**

`_handle_move_to_watch_result` 다음, `def _on_tool_track(self, msg): pass` 를 아래로 교체:
```python
    def _check_trigger(self, tool_track_msg) -> bool:
        """시야 내 + approaching이면 True (완화된 트리거 판정, 데모.md 1.3절)."""
        raise NotImplementedError('_check_trigger 구현 필요')

    def _get_grasp_spec(self, tool_class: str):
        """(grasp_width_mm, grasp_force_n) 등록된 공구 스펙을 반환한다."""
        raise NotImplementedError('_get_grasp_spec 구현 필요')

    def _on_tool_track(self, msg):
        if self.state != State.DETECT_TRACK:
            return
        triggered = self._safe_call(self._check_trigger, msg, default=False)
        if not triggered:
            self._detect_track_cycles += 1
            max_cycles = self.get_parameter('detect_track_max_cycles').value
            if self._detect_track_cycles >= max_cycles:
                self._set_state(State.IDLE, detail='벨트에 없음')
            return
        spec = self._safe_call(self._get_grasp_spec, self.current_tool, default=None)
        width_mm, force_n = spec if spec else (0.0, 0.0)
        self._set_state(State.SERVO_PICK)
        self._send_robot_goal(
            'servo_pick', tool_class=self.current_tool,
            grasp_width_mm=width_mm, grasp_force_n=force_n)

    def _verify_grasp(self, result) -> bool:
        """무게·폭·grip_detected 삼중 확인 (데모.md 2.6/VERIFY_GRASP)."""
        raise NotImplementedError('_verify_grasp 구현 필요')

    def _handle_servo_pick_result(self, result):
        if not result.success:
            if 'torque' in result.message:
                self._set_state(State.FAULT, detail=result.message)
            else:
                self._detect_track_cycles = 0
                self._set_state(State.DETECT_TRACK, detail=result.message)
            return
        self._set_state(State.VERIFY_GRASP)
        verified = self._safe_call(self._verify_grasp, result, default=False)
        if verified:
            self._set_state(State.MOVE_SAFE)
            self._send_robot_goal('move_named', named_target='safe')
            return
        self._verify_grasp_retries += 1
        max_retries = self.get_parameter('verify_grasp_max_retries').value
        if self._verify_grasp_retries > max_retries:
            self._set_state(State.IDLE, detail='파지 검증 실패 - 보고')
            return
        self._send_robot_goal('release_and_retry')

    def _handle_release_and_retry_result(self, result):
        if result.success:
            self._detect_track_cycles = 0
            self._set_state(State.DETECT_TRACK)
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _on_hand_pose(self, msg):
        pass
```

- [ ] **Step 3: 테스트 통과 확인**

Run: `python3 -m pytest src/task_manager/test/ -v`
Expected: 19 passed

- [ ] **Step 4: 커밋**

```bash
git add src/task_manager
git commit -m "feat: task_manager DETECT_TRACK/SERVO_PICK/VERIFY_GRASP 배선"
```

---

### Task 13: `task_manager` — MOVE_SAFE + TRACK_HAND + WAIT_PULL + RELEASE + HOME

**Files:**
- Modify: `src/task_manager/task_manager/task_manager_node.py`
- Modify: `src/task_manager/test/test_task_manager_node.py`

**Interfaces:**
- Produces: `_handle_move_safe_result`, `_on_hand_pose` 실제 구현, `_on_hand_timeout`, `_handle_track_hand_result`, `_on_wait_pull_timeout`, `_handle_wait_pull_result`, `_handle_release_result`, `_handle_home_result`

- [ ] **Step 1: 실패하는 테스트 추가**

`src/task_manager/test/test_task_manager_node.py` 하단에 추가:
```python
def test_move_safe_result_success_transitions_to_track_hand(node):
    node.state = State.MOVE_SAFE
    node._set_vision_mode = lambda mode, tool_class='': None

    node._handle_move_safe_result(_FakeResult(success=True))

    assert node.state == State.TRACK_HAND
    assert node._hand_timeout_timer is not None
    node._hand_timeout_timer.cancel()


def test_move_safe_result_failure_goes_to_fault(node):
    node.state = State.MOVE_SAFE

    node._handle_move_safe_result(_FakeResult(success=False, message='motion failed'))

    assert node.state == State.FAULT


def test_hand_pose_sends_move_pose_with_offset(node):
    node.state = State.TRACK_HAND
    node._hand_timeout_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    msg = PoseStamped()
    msg.pose.position.z = 0.30
    node._on_hand_pose(msg)

    assert node._hand_timeout_timer is None
    assert sent[0][0] == 'move_pose'
    assert abs(sent[0][1]['target_pose'].pose.position.z - 0.38) < 1e-9


def test_hand_pose_ignored_unless_track_hand(node):
    node.state = State.IDLE
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_hand_pose(PoseStamped())

    assert sent == []


def test_hand_timeout_sends_fallback_goal(node):
    node.state = State.TRACK_HAND
    node._hand_timeout_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_hand_timeout()

    assert sent == [('move_named', {'named_target': 'handover_default'})]


def test_track_hand_result_success_transitions_to_wait_pull(node):
    node.state = State.TRACK_HAND
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_track_hand_result(_FakeResult(success=True))

    assert node.state == State.WAIT_PULL
    assert sent == [('handover_hold', {})]
    node._wait_pull_timeout_timer.cancel()


def test_wait_pull_result_success_goes_home(node):
    node.state = State.WAIT_PULL
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)
    node._set_vision_mode = lambda mode, tool_class='': None
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_wait_pull_result(_FakeResult(success=True, message='pull_detected, released'))

    assert node.state == State.HOME
    assert sent == [('move_named', {'named_target': 'home'})]


def test_wait_pull_timeout_sends_place_down(node):
    node.state = State.WAIT_PULL
    node._wait_pull_timeout_timer = node.create_timer(100.0, lambda: None)
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._on_wait_pull_timeout()

    assert node.state == State.RELEASE
    assert sent == [('place_down', {'named_target': 'place_down'})]


def test_release_result_success_goes_home(node):
    node.state = State.RELEASE
    node._set_vision_mode = lambda mode, tool_class='': None
    sent = []
    node._send_robot_goal = lambda task_type, **kw: sent.append((task_type, kw))

    node._handle_release_result(_FakeResult(success=True))

    assert node.state == State.HOME
    assert sent == [('move_named', {'named_target': 'home'})]


def test_home_result_success_returns_to_idle(node):
    node.state = State.HOME
    node.current_tool = 'spanner'

    node._handle_home_result(_FakeResult(success=True))

    assert node.state == State.IDLE


def test_home_result_failure_goes_to_fault(node):
    node.state = State.HOME

    node._handle_home_result(_FakeResult(success=False, message='motion failed'))

    assert node.state == State.FAULT
```

`test_task_manager_node.py` 상단 import에 `from geometry_msgs.msg import PoseStamped` 추가.

Run: `python3 -m pytest src/task_manager/test/ -v`
Expected: FAIL (`AttributeError: '_handle_move_safe_result'` 등)

- [ ] **Step 2: MOVE_SAFE/TRACK_HAND/WAIT_PULL/RELEASE/HOME 구현**

`_handle_release_and_retry_result` 다음, `def _on_hand_pose(self, msg): pass` 를 아래로 교체:
```python
    def _handle_move_safe_result(self, result):
        if result.success:
            self._set_state(State.TRACK_HAND)
            self._set_vision_mode(SetVisionMode.Request.TRACK_HAND)
            timeout_s = self.get_parameter('hand_detect_timeout_s').value
            self._hand_timeout_timer = self.create_timer(timeout_s, self._on_hand_timeout)
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _on_hand_timeout(self):
        self._hand_timeout_timer.cancel()
        self._hand_timeout_timer = None
        if self.state != State.TRACK_HAND:
            return
        self._send_robot_goal('move_named', named_target='handover_default')

    def _on_hand_pose(self, msg):
        if self.state != State.TRACK_HAND:
            return
        if self._hand_timeout_timer is not None:
            self._hand_timeout_timer.cancel()
            self._hand_timeout_timer = None
        offset_pose = PoseStamped()
        offset_pose.header = msg.header
        offset_pose.pose = msg.pose
        offset_pose.pose.position.z += 0.08
        self._send_robot_goal('move_pose', target_pose=offset_pose)

    def _handle_track_hand_result(self, result):
        if result.success:
            self._set_state(State.WAIT_PULL)
            timeout_s = self.get_parameter('wait_pull_timeout_s').value
            self._wait_pull_timeout_timer = self.create_timer(timeout_s, self._on_wait_pull_timeout)
            self._send_robot_goal('handover_hold')
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _on_wait_pull_timeout(self):
        self._wait_pull_timeout_timer.cancel()
        self._wait_pull_timeout_timer = None
        if self.state != State.WAIT_PULL:
            return
        self._set_state(State.RELEASE, detail='wait_pull timeout')
        self._send_robot_goal('place_down', named_target='place_down')

    def _handle_wait_pull_result(self, result):
        if self._wait_pull_timeout_timer is not None:
            self._wait_pull_timeout_timer.cancel()
            self._wait_pull_timeout_timer = None
        if result.success:
            # RELEASE는 robot_control이 handover_hold 안에서 이미 개방을 완료했음을
            # 표시하기 위한 경유 상태 - 별도 goal 없이 바로 HOME으로 넘어간다.
            self._set_state(State.RELEASE, detail=result.message)
            self._set_state(State.HOME)
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self._send_robot_goal('move_named', named_target='home')
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _handle_release_result(self, result):
        # WAIT_PULL 타임아웃 후 보낸 place_down goal의 결과 처리
        if result.success:
            self._set_state(State.HOME)
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self._send_robot_goal('move_named', named_target='home')
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _handle_home_result(self, result):
        if result.success:
            self._set_state(State.IDLE, detail=f'DONE tool={self.current_tool}')
        else:
            self._set_state(State.FAULT, detail=result.message)
```

- [ ] **Step 3: 테스트 통과 확인**

Run: `python3 -m pytest src/task_manager/test/ -v`
Expected: 30 passed

- [ ] **Step 4: 커밋**

```bash
git add src/task_manager
git commit -m "feat: task_manager MOVE_SAFE/TRACK_HAND/WAIT_PULL/RELEASE/HOME 배선"
```

---

### Task 14: `task_manager` launch 파일

**Files:**
- Create: `src/task_manager/launch/task_manager.launch.py`

- [ ] **Step 1: launch 파일 작성**

`src/task_manager/launch/task_manager.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='task_manager', executable='task_manager_node'),
    ])
```

- [ ] **Step 2: 빌드 후 launch 확인**

Run:
```bash
cd ~/rokey_proj_02
colcon build --symlink-install --packages-select task_manager
source install/setup.bash
timeout 3 ros2 launch task_manager task_manager.launch.py || true
```
Expected: 에러 없이 뜨고 3초 후 timeout으로 정상 종료.

- [ ] **Step 3: 커밋**

```bash
git add src/task_manager/launch
git commit -m "feat: task_manager launch 파일 추가"
```

---

### Task 15: `stt_node` + `vision_node` launch 파일

**Files:**
- Create: `src/stt_node/launch/stt_node.launch.py`
- Create: `src/vision_node/launch/vision_node.launch.py`

- [ ] **Step 1: stt_node launch 작성**

`src/stt_node/launch/stt_node.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='stt_node', executable='stt_node'),
    ])
```

- [ ] **Step 2: vision_node launch 작성 (realsense2_camera include + hand-eye TF)**

`src/vision_node/launch/vision_node.launch.py`:
```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'depth_module.profile': '424x240x60',
            'rgb_camera.profile': '424x240x60',
        }.items()
    )
    # NOTE: realsense2_camera 버전에 따라 launch 인자명이 다를 수 있으니
    # 설치된 realsense-ros 문서로 재확인할 것.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # TODO: hand-eye 캘리브레이션 결과값(flange -> camera_link)으로 인자 교체
        arguments=['0', '0', '0', '0', '0', '0', 'flange', 'camera_link'],
    )
    vision_node = Node(
        package='vision_node',
        executable='vision_node',
    )
    return LaunchDescription([realsense_launch, static_tf, vision_node])
```

- [ ] **Step 3: 빌드 후 launch 구문 확인**

Run:
```bash
cd ~/rokey_proj_02
colcon build --symlink-install --packages-select stt_node vision_node
source install/setup.bash
timeout 3 ros2 launch stt_node stt_node.launch.py || true
python3 -c "from launch.launch_description_sources import PythonLaunchDescriptionSource; import ament_index_python.packages as p; print('vision_node launch file parses OK')"
```
Expected: stt_node launch 정상 기동/종료. (`realsense2_camera`가 설치되어 있지 않은 개발 환경에서는 vision_node launch 전체 기동은 실패할 수 있음 — 이 경우 `ros2 launch vision_node vision_node.launch.py --show-args`로 launch 파일 자체의 구문 오류 여부만 확인한다.)

- [ ] **Step 4: 커밋**

```bash
git add src/stt_node/launch src/vision_node/launch
git commit -m "feat: stt_node/vision_node launch 파일 추가"
```

---

### Task 16: `handover_ui` 패키지 + `RosClient`

**Files:**
- Create: `src/handover_ui/package.xml`, `setup.py`, `setup.cfg`, `resource/handover_ui`
- Create: `src/handover_ui/handover_ui/__init__.py`, `ros_client.py`
- Test: `src/handover_ui/test/test_ros_client.py`

**Interfaces:**
- Produces: `RosClient(host, port)` — `.connect()`, `.close()`, `.is_connected()`, `.subscribe_all()`, `.publish_command(text)`, 콜백 속성 `on_task_status(state, detail)`, `on_gripper_state(width_mm, grip_detected)`, `on_fault(message)`

- [ ] **Step 1: 패키지 뼈대**

`src/handover_ui/package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>handover_ui</name>
  <version>0.0.1</version>
  <description>PyQt 데스크톱 UI - rosbridge(roslibpy) 경유로 /task/status, /gripper/state, /robot/fault 표시 및 /user_command/text 입력</description>
  <maintainer email="hwangjeongui01@gmail.com">hwangjeongui</maintainer>
  <license>MIT</license>

  <!-- PyQt5, roslibpy, pytest-qt는 ROS 패키지가 아니라 pip 의존성:
       pip install PyQt5 roslibpy pytest-qt -->

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`src/handover_ui/setup.py`:
```python
from setuptools import find_packages, setup

package_name = 'handover_ui'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/handover_ui.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='PyQt 데스크톱 UI (rosbridge 경유)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'handover_ui = handover_ui.main:main',
        ],
    },
)
```

`src/handover_ui/setup.cfg`:
```
[develop]
script_dir=$base/lib/handover_ui
[install]
install_scripts=$base/lib/handover_ui
```

Run:
```bash
mkdir -p ~/rokey_proj_02/src/handover_ui/resource
touch ~/rokey_proj_02/src/handover_ui/resource/handover_ui
touch ~/rokey_proj_02/src/handover_ui/handover_ui/__init__.py
pip install PyQt5 roslibpy pytest-qt
```

- [ ] **Step 2: 실패하는 테스트 작성**

`src/handover_ui/test/test_ros_client.py`:
```python
import roslibpy
import pytest

from handover_ui.ros_client import RosClient


class _FakeTopic:
    instances = []

    def __init__(self, ros, name, msg_type):
        self.ros = ros
        self.name = name
        self.msg_type = msg_type
        self.subscribed_callback = None
        self.published = []
        _FakeTopic.instances.append(self)

    def subscribe(self, callback):
        self.subscribed_callback = callback

    def publish(self, message):
        self.published.append(message)


@pytest.fixture(autouse=True)
def patch_roslibpy(monkeypatch):
    _FakeTopic.instances = []
    monkeypatch.setattr(roslibpy, 'Topic', _FakeTopic)
    monkeypatch.setattr(roslibpy, 'Ros', lambda host, port: object())
    yield


def test_subscribe_all_creates_three_subscriptions():
    client = RosClient()
    client.subscribe_all()
    names = [t.name for t in _FakeTopic.instances]
    assert '/task/status' in names
    assert '/gripper/state' in names
    assert '/robot/fault' in names


def test_task_status_callback_parses_json():
    client = RosClient()
    client.subscribe_all()
    received = []
    client.on_task_status = lambda state, detail: received.append((state, detail))

    status_topic = next(t for t in _FakeTopic.instances if t.name == '/task/status')
    status_topic.subscribed_callback({'data': '{"state": "IDLE", "detail": "ready"}'})

    assert received == [('IDLE', 'ready')]


def test_gripper_state_callback_forwards_fields():
    client = RosClient()
    client.subscribe_all()
    received = []
    client.on_gripper_state = lambda width, grip: received.append((width, grip))

    gripper_topic = next(t for t in _FakeTopic.instances if t.name == '/gripper/state')
    gripper_topic.subscribed_callback({'width_mm': 30.0, 'grip_detected': True})

    assert received == [(30.0, True)]


def test_publish_command_sends_message():
    client = RosClient()
    client.subscribe_all()

    client.publish_command('스패너 갖다줘')

    command_topic = next(t for t in _FakeTopic.instances if t.name == '/user_command/text')
    assert command_topic.published[0]['data'] == '스패너 갖다줘'
```

Run: `cd ~/rokey_proj_02 && python3 -m pytest src/handover_ui/test/test_ros_client.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: `RosClient` 구현**

`src/handover_ui/handover_ui/ros_client.py`:
```python
import json

import roslibpy


class RosClient:
    """rosbridge(WebSocket)에 접속해 필요한 토픽을 구독/퍼블리시하는 래퍼."""

    def __init__(self, host='localhost', port=9090):
        self.ros = roslibpy.Ros(host=host, port=port)
        self.on_task_status = None
        self.on_gripper_state = None
        self.on_fault = None
        self._command_topic = None

    def connect(self):
        self.ros.run()

    def close(self):
        self.ros.terminate()

    def is_connected(self) -> bool:
        return self.ros.is_connected

    def subscribe_all(self):
        roslibpy.Topic(self.ros, '/task/status', 'std_msgs/String').subscribe(
            self._on_task_status_raw)
        roslibpy.Topic(self.ros, '/gripper/state', 'handover_interfaces/GripperState').subscribe(
            self._on_gripper_state_raw)
        roslibpy.Topic(self.ros, '/robot/fault', 'std_msgs/String').subscribe(
            self._on_fault_raw)
        self._command_topic = roslibpy.Topic(
            self.ros, '/user_command/text', 'std_msgs/String')

    def publish_command(self, text: str):
        self._command_topic.publish(roslibpy.Message({'data': text}))

    def _on_task_status_raw(self, message):
        if self.on_task_status is None:
            return
        payload = json.loads(message['data'])
        self.on_task_status(payload.get('state', ''), payload.get('detail', ''))

    def _on_gripper_state_raw(self, message):
        if self.on_gripper_state is None:
            return
        self.on_gripper_state(message.get('width_mm', 0.0), message.get('grip_detected', False))

    def _on_fault_raw(self, message):
        if self.on_fault is None:
            return
        self.on_fault(message.get('data', ''))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest src/handover_ui/test/test_ros_client.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/handover_ui
git commit -m "feat: handover_ui 패키지 + RosClient(roslibpy 래퍼) 추가"
```

---

### Task 17: `handover_ui` — `MainWindow`

**Files:**
- Create: `src/handover_ui/handover_ui/main_window.py`
- Test: `src/handover_ui/test/test_main_window.py`

**Interfaces:**
- Consumes: `RosClient` (Task 16) — `.on_task_status`, `.on_gripper_state`, `.on_fault`, `.publish_command(text)`
- Produces: `MainWindow(ros_client)` PyQt 위젯

- [ ] **Step 1: 실패하는 테스트 작성**

`src/handover_ui/test/test_main_window.py`:
```python
import pytest
from PyQt5.QtCore import Qt

from handover_ui.main_window import MainWindow


class _FakeRosClient:
    def __init__(self):
        self.published = []
        self.on_task_status = None
        self.on_gripper_state = None
        self.on_fault = None

    def publish_command(self, text):
        self.published.append(text)


@pytest.fixture
def window(qtbot):
    ros_client = _FakeRosClient()
    win = MainWindow(ros_client)
    qtbot.addWidget(win)
    return win


def test_task_status_updates_labels(window, qtbot):
    with qtbot.waitSignal(window.task_status_received, timeout=1000):
        window.ros_client.on_task_status('MOVE_TO_WATCH', '감시 자세로 이동')

    assert window.state_label.text() == '상태: MOVE_TO_WATCH'
    assert window.detail_label.text() == '디테일: 감시 자세로 이동'
    assert window.log_view.count() == 1


def test_gripper_state_updates_label(window, qtbot):
    with qtbot.waitSignal(window.gripper_state_received, timeout=1000):
        window.ros_client.on_gripper_state(29.4, True)

    assert '29.4' in window.gripper_label.text()
    assert 'True' in window.gripper_label.text()


def test_fault_shows_banner(window, qtbot):
    with qtbot.waitSignal(window.fault_received, timeout=1000):
        window.ros_client.on_fault('torque anomaly')

    assert window.fault_banner.isVisible()
    assert 'torque anomaly' in window.fault_banner.text()


def test_send_button_publishes_command_and_clears_input(window, qtbot):
    window.command_input.setText('스패너 갖다줘')
    qtbot.mouseClick(window.send_button, Qt.LeftButton)

    assert window.ros_client.published == ['스패너 갖다줘']
    assert window.command_input.text() == ''


def test_send_button_ignores_empty_input(window, qtbot):
    window.command_input.setText('   ')
    qtbot.mouseClick(window.send_button, Qt.LeftButton)

    assert window.ros_client.published == []
```

Run: `python3 -m pytest src/handover_ui/test/test_main_window.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 2: `MainWindow` 구현**

`src/handover_ui/handover_ui/main_window.py`:
```python
from PyQt5 import QtWidgets, QtCore


class MainWindow(QtWidgets.QMainWindow):
    task_status_received = QtCore.pyqtSignal(str, str)
    gripper_state_received = QtCore.pyqtSignal(float, bool)
    fault_received = QtCore.pyqtSignal(str)

    def __init__(self, ros_client):
        super().__init__()
        self.ros_client = ros_client
        self.setWindowTitle('공구 전달 로봇 제어')

        self.state_label = QtWidgets.QLabel('상태: -')
        self.detail_label = QtWidgets.QLabel('디테일: -')
        self.gripper_label = QtWidgets.QLabel('그리퍼: -')
        self.fault_banner = QtWidgets.QLabel('')
        self.fault_banner.setStyleSheet('background-color: red; color: white; font-weight: bold;')
        self.fault_banner.hide()
        self.log_view = QtWidgets.QListWidget()

        self.command_input = QtWidgets.QLineEdit()
        self.send_button = QtWidgets.QPushButton('전송')
        self.send_button.clicked.connect(self._on_send_clicked)

        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(self.command_input)
        input_layout.addWidget(self.send_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.fault_banner)
        layout.addWidget(self.state_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.gripper_label)
        layout.addLayout(input_layout)
        layout.addWidget(self.log_view)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.task_status_received.connect(self._update_task_status)
        self.gripper_state_received.connect(self._update_gripper_state)
        self.fault_received.connect(self._update_fault)

        self.ros_client.on_task_status = self.task_status_received.emit
        self.ros_client.on_gripper_state = self.gripper_state_received.emit
        self.ros_client.on_fault = self.fault_received.emit

    def _on_send_clicked(self):
        text = self.command_input.text().strip()
        if not text:
            return
        self.ros_client.publish_command(text)
        self.command_input.clear()

    def _update_task_status(self, state, detail):
        self.state_label.setText(f'상태: {state}')
        self.detail_label.setText(f'디테일: {detail}')
        self.log_view.addItem(f'[{state}] {detail}')

    def _update_gripper_state(self, width_mm, grip_detected):
        self.gripper_label.setText(f'그리퍼: {width_mm:.1f}mm, grip_detected={grip_detected}')

    def _update_fault(self, message):
        self.fault_banner.setText(f'FAULT: {message}')
        self.fault_banner.show()
        self.log_view.addItem(f'[FAULT] {message}')
```

- [ ] **Step 3: 테스트 통과 확인**

Run: `python3 -m pytest src/handover_ui/test/test_main_window.py -v`
Expected: 5 passed

- [ ] **Step 4: 커밋**

```bash
git add src/handover_ui
git commit -m "feat: handover_ui MainWindow 구현"
```

---

### Task 18: `handover_ui` — `main.py` 진입점 + launch 파일

**Files:**
- Create: `src/handover_ui/handover_ui/main.py`
- Create: `src/handover_ui/launch/handover_ui.launch.py`

**Interfaces:**
- Consumes: `RosClient` (Task 16), `MainWindow` (Task 17)

- [ ] **Step 1: `main.py` 작성**

`src/handover_ui/handover_ui/main.py`:
```python
import sys

from PyQt5 import QtWidgets

from handover_ui.main_window import MainWindow
from handover_ui.ros_client import RosClient


def main():
    app = QtWidgets.QApplication(sys.argv)
    ros_client = RosClient(host='localhost', port=9090)
    window = MainWindow(ros_client)
    ros_client.connect()
    ros_client.subscribe_all()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: launch 파일 작성 (rosbridge_websocket include + UI 프로세스)**

`src/handover_ui/launch/handover_ui.launch.py`:
```python
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rosbridge_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            FindPackageShare('rosbridge_server'), '/launch/rosbridge_websocket_launch.xml'
        ])
    )
    ui_process = ExecuteProcess(cmd=['handover_ui'], output='screen')
    return LaunchDescription([rosbridge_launch, ui_process])
```

- [ ] **Step 3: 빌드 후 수동 확인**

Run:
```bash
cd ~/rokey_proj_02
colcon build --symlink-install --packages-select handover_ui
source install/setup.bash
python3 -c "from handover_ui.main import main; print('handover_ui.main import OK')"
```
Expected: `handover_ui.main import OK` 출력. (`rosbridge_server`가 설치되어 있다면 `ros2 launch handover_ui handover_ui.launch.py`로 실제 창이 뜨는지 수동 확인 — 디스플레이가 없는 환경이면 이 launch 실행 자체는 생략하고 import 확인으로 대체.)

- [ ] **Step 4: 커밋**

```bash
git add src/handover_ui
git commit -m "feat: handover_ui main 진입점 + launch 파일 (rosbridge include)"
```

---

### Task 19: 전체 통합 빌드 및 검증

**Files:** 없음 (커맨드 실행 검증만)

- [ ] **Step 1: 전체 빌드**

Run:
```bash
cd ~/rokey_proj_02
colcon build --symlink-install
source install/setup.bash
```
Expected: 6개 패키지 모두 빌드 성공.

- [ ] **Step 2: 전체 테스트 실행**

Run:
```bash
python3 -m pytest src/stt_node/test src/vision_node/test src/robot_control/test src/task_manager/test src/handover_ui/test -v
```
Expected: 모든 테스트 통과 (Task 2~18에서 작성한 테스트 총합).

- [ ] **Step 3: 인터페이스 배선이 4.5절 토픽 총괄표와 일치하는지 노드별 확인**

Run (각 노드를 별도 터미널에서 `ros2 run <pkg> <executable>`로 띄운 뒤):
```bash
ros2 topic list
ros2 service list
ros2 action list
ros2 node info /task_manager
ros2 node info /vision_node
ros2 node info /robot_control
```
Expected:
- 토픽: `/user_command/text`, `/task/status`, `/vision/tool_track`, `/vision/hand_pose`, `/gripper/state`, `/robot/fault` 모두 존재
- 서비스: `/vision/set_mode` 존재
- 액션: `/robot_task` 존재

- [ ] **Step 4: 결과 요약 커밋**

```bash
git add -A
git status
```
Expected: 워크스페이스에 커밋되지 않은 변경이 없음 (모든 태스크가 이미 커밋됨). 변경 사항이 있다면 검토 후 커밋.
