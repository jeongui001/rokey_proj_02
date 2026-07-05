import base64
import time

import roslibpy
import pytest

from operator_gui.ros_client import DEFAULT_CAMERA_TOPIC, RosClient


class _FakeTopic:
    instances = []
    raise_on_unsubscribe = False
    raise_on_unadvertise = False
    fail_subscribe_for = None  # 이 이름의 토픽에서 subscribe() 호출 시 예외 발생

    def __init__(self, ros, name, msg_type):
        self.ros = ros
        self.name = name
        self.msg_type = msg_type
        self.subscribed_callback = None
        self.published = []
        self.unsubscribed = False
        self.unadvertised = False
        _FakeTopic.instances.append(self)

    def subscribe(self, callback):
        if _FakeTopic.fail_subscribe_for == self.name:
            raise RuntimeError(f'subscribe failed for {self.name} (simulated)')
        self.subscribed_callback = callback

    def publish(self, message):
        self.published.append(message)

    def unsubscribe(self):
        if _FakeTopic.raise_on_unsubscribe:
            raise RuntimeError('unsubscribe failed (simulated)')
        self.unsubscribed = True
        self.subscribed_callback = None

    def unadvertise(self):
        if _FakeTopic.raise_on_unadvertise:
            raise RuntimeError('unadvertise failed (simulated)')
        self.unadvertised = True


class _FakeRos:
    def __init__(self, host=None, port=None):
        self.host = host
        self.port = port
        self.is_connected = False
        self.ready_callbacks = []
        self.event_callbacks = {}
        self.run_calls = []
        self.terminate_called = False

    def on_ready(self, callback, run_in_thread=True):
        self.ready_callbacks.append(callback)

    def on(self, event, callback):
        self.event_callbacks[event] = callback

    def run(self, timeout=None):
        self.run_calls.append(timeout)
        self.is_connected = True
        for cb in list(self.ready_callbacks):
            cb()

    def terminate(self):
        self.terminate_called = True
        self.is_connected = False


@pytest.fixture(autouse=True)
def patch_roslibpy(monkeypatch):
    _FakeTopic.instances = []
    _FakeTopic.raise_on_unsubscribe = False
    _FakeTopic.raise_on_unadvertise = False
    _FakeTopic.fail_subscribe_for = None
    monkeypatch.setattr(roslibpy, 'Topic', _FakeTopic)
    monkeypatch.setattr(roslibpy, 'Ros', _FakeRos)
    yield


def _topic(name):
    return next(t for t in _FakeTopic.instances if t.name == name)


# ---- 구독 ----

def test_subscribe_all_creates_expected_subscriptions():
    client = RosClient()
    client.subscribe_all()

    names = [t.name for t in _FakeTopic.instances]
    assert '/task/status' in names
    assert '/gripper/state' in names
    assert '/robot/fault' in names
    assert DEFAULT_CAMERA_TOPIC in names
    assert '/user_command/text' in names


# ---- 중복 구독 방지 ----

def test_subscribe_all_called_twice_does_not_duplicate_topics():
    client = RosClient()

    client.subscribe_all()
    client.subscribe_all()  # on_ready가 두 번 호출되는 상황을 흉내낸다

    status_topics = [t for t in _FakeTopic.instances if t.name == '/task/status']
    assert len(status_topics) == 1
    command_topics = [t for t in _FakeTopic.instances if t.name == '/user_command/text']
    assert len(command_topics) == 1


def test_on_ready_fired_twice_does_not_duplicate_topics():
    client = RosClient()

    client.ros.ready_callbacks[0]()
    client.ros.ready_callbacks[0]()

    status_topics = [t for t in _FakeTopic.instances if t.name == '/task/status']
    assert len(status_topics) == 1


def test_disconnect_then_reconnect_resubscribes_exactly_once():
    client = RosClient()

    client.ros.ready_callbacks[0]()  # 최초 연결
    first_status_topic = _topic('/task/status')

    client.ros.event_callbacks['close']()  # 연결 끊김 시뮬레이션
    assert client._subscribed_topics == []
    assert first_status_topic.unsubscribed is True

    client.ros.ready_callbacks[0]()  # 재연결
    status_topics = [t for t in _FakeTopic.instances if t.name == '/task/status']
    assert len(status_topics) == 2  # 이전 것 + 새로 구독한 것, 총 두 개의 인스턴스만 존재
    assert client._subscribed_topics  # 새로 구독된 상태
    assert len([t for t in client._subscribed_topics if t.name == '/task/status']) == 1


@pytest.mark.parametrize('event', ['close', 'error'])
def test_reconnect_after_close_or_error_resubscribes_topics_exactly_once(event):
    # error 이벤트도 close와 동일하게 유령 구독을 정리해야 한다.
    client = RosClient()

    client.ros.ready_callbacks[0]()
    client.ros.event_callbacks[event]()
    assert client._subscribed_topics == []
    assert client._command_topic is None

    client.ros.ready_callbacks[0]()
    status_topics = [t for t in _FakeTopic.instances if t.name == '/task/status']
    assert len(status_topics) == 2
    assert len([t for t in client._subscribed_topics if t.name == '/task/status']) == 1


def test_close_unsubscribes_topics_and_unadvertises_command_topic():
    client = RosClient()
    client.subscribe_all()
    status_topic = _topic('/task/status')
    command_topic = _topic('/user_command/text')

    client.close()

    assert status_topic.unsubscribed is True
    assert command_topic.unadvertised is True
    assert client._subscribed_topics == []
    assert client._command_topic is None


def test_close_survives_unsubscribe_failure():
    client = RosClient()
    client.subscribe_all()
    _FakeTopic.raise_on_unsubscribe = True
    _FakeTopic.raise_on_unadvertise = True

    client.close()  # 예외가 전파되어 UI가 죽으면 안 된다

    assert client._subscribed_topics == []
    assert client.ros.terminate_called is True


def test_on_close_event_survives_unsubscribe_failure():
    client = RosClient()
    client.subscribe_all()
    _FakeTopic.raise_on_unsubscribe = True

    client.ros.event_callbacks['close']()  # 예외 없이 처리되어야 한다

    assert client._subscribed_topics == []


def test_on_error_event_tears_down_subscriptions():
    client = RosClient()
    client.subscribe_all()
    status_topic = _topic('/task/status')
    command_topic = _topic('/user_command/text')
    states = []
    client.on_connection_changed = states.append

    client.ros.event_callbacks['error']()

    assert status_topic.unsubscribed is True
    assert command_topic.unadvertised is True
    assert client._subscribed_topics == []
    assert client._command_topic is None
    assert states == [False]


def test_on_error_event_survives_unsubscribe_failure():
    client = RosClient()
    client.subscribe_all()
    _FakeTopic.raise_on_unsubscribe = True

    client.ros.event_callbacks['error']()  # 예외 없이 처리되어야 한다

    assert client._subscribed_topics == []


# ---- 구독 도중 일부 실패 시 정리 ----

def test_subscribe_all_partial_failure_cleans_up_and_resets_state():
    client = RosClient()
    _FakeTopic.fail_subscribe_for = '/robot/fault'  # 3번째 토픽에서 실패시킨다

    result = client.subscribe_all()

    assert result is False
    assert client._subscribed_topics == []
    assert client._command_topic is None
    assert client._subscriptions_ready is False
    # 실패 전에 이미 구독됐던 토픽들(/task/status, /gripper/state)은 정리(unsubscribe)됐어야 한다.
    earlier_topics = [t for t in _FakeTopic.instances
                       if t.name in ('/task/status', '/gripper/state')]
    assert len(earlier_topics) == 2
    assert all(t.unsubscribed for t in earlier_topics)
    # 실패한 토픽 자체나 그 이후 토픽(카메라)은 생성되지 않는다.
    assert not any(t.name == DEFAULT_CAMERA_TOPIC for t in _FakeTopic.instances)


def test_subscribe_all_can_retry_successfully_after_partial_failure():
    client = RosClient()
    _FakeTopic.fail_subscribe_for = '/robot/fault'

    client.subscribe_all()
    assert client._subscribed_topics == []

    _FakeTopic.fail_subscribe_for = None
    client.subscribe_all()

    names = [t.name for t in client._subscribed_topics]
    assert '/task/status' in names
    assert '/gripper/state' in names
    assert '/robot/fault' in names
    assert DEFAULT_CAMERA_TOPIC in names
    assert client._command_topic is not None


def test_task_status_callback_parses_four_fields():
    client = RosClient()
    client.subscribe_all()
    received = []
    client.on_task_status = lambda *args: received.append(args)

    _topic('/task/status').subscribed_callback({
        'data': '{"state": "IDLE", "detail": "ready", '
                '"operation_mode": "AUTO", "safety_state": "NORMAL"}'
    })

    assert received == [('IDLE', 'ready', 'AUTO', 'NORMAL')]


def test_gripper_state_callback_forwards_fields():
    client = RosClient()
    client.subscribe_all()
    received = []
    client.on_gripper_state = lambda width, grip: received.append((width, grip))

    _topic('/gripper/state').subscribed_callback({'width_mm': 30.0, 'grip_detected': True})

    assert received == [(30.0, True)]


def test_camera_image_callback_decodes_base64_to_bytes():
    client = RosClient()
    client.subscribe_all()
    received = []
    client.on_camera_image = received.append

    raw_bytes = b'\x89PNG\r\n\x1a\nfake-image-data'
    encoded = base64.b64encode(raw_bytes).decode('ascii')
    _topic(DEFAULT_CAMERA_TOPIC).subscribed_callback({'data': encoded, 'format': 'jpeg'})

    assert received == [raw_bytes]


def test_camera_image_callback_ignores_missing_data():
    client = RosClient()
    client.subscribe_all()
    received = []
    client.on_camera_image = received.append

    _topic(DEFAULT_CAMERA_TOPIC).subscribed_callback({'format': 'jpeg'})

    assert received == []


# ---- 카메라 토픽 설정 ----

def test_camera_topic_defaults_when_no_env_or_arg(monkeypatch):
    monkeypatch.delenv('OPERATOR_GUI_CAMERA_TOPIC', raising=False)
    client = RosClient()
    assert client.camera_topic == DEFAULT_CAMERA_TOPIC


def test_camera_topic_from_env_var(monkeypatch):
    monkeypatch.setenv('OPERATOR_GUI_CAMERA_TOPIC', '/custom/camera/compressed')
    client = RosClient()
    assert client.camera_topic == '/custom/camera/compressed'


def test_camera_topic_constructor_arg_overrides_env(monkeypatch):
    monkeypatch.setenv('OPERATOR_GUI_CAMERA_TOPIC', '/env/camera')
    client = RosClient(camera_topic='/explicit/camera')
    assert client.camera_topic == '/explicit/camera'


# ---- 명령 전송 ----

def test_publish_command_sends_message_when_connected():
    client = RosClient()
    client.subscribe_all()
    client.ros.is_connected = True

    sent = client.publish_command('스패너 갖다줘')

    assert sent is True
    assert _topic('/user_command/text').published[0]['data'] == '스패너 갖다줘'


def test_publish_command_blocked_when_not_connected():
    client = RosClient()
    client.subscribe_all()
    client.ros.is_connected = False

    sent = client.publish_command('스패너 갖다줘')

    assert sent is False
    assert _topic('/user_command/text').published == []


@pytest.mark.parametrize('text', ['', '   ', None])
def test_publish_command_ignores_empty_text(text):
    client = RosClient()
    client.subscribe_all()
    client.ros.is_connected = True

    assert client.publish_command(text) is False
    assert _topic('/user_command/text').published == []


# ---- 연결 / 재연결 ----

def test_connect_runs_in_background_and_does_not_block_caller():
    client = RosClient()

    def slow_run(timeout=None):
        time.sleep(0.3)
        client.ros.is_connected = True

    client.ros.run = slow_run

    start = time.monotonic()
    client.connect()
    elapsed = time.monotonic() - start

    assert elapsed < 0.1


def test_on_ready_auto_subscribes_and_reports_connected():
    client = RosClient()
    states = []
    client.on_connection_changed = states.append

    client.ros.ready_callbacks[0]()

    names = [t.name for t in _FakeTopic.instances]
    assert '/task/status' in names
    assert states == [True]


def test_ensure_connected_skips_when_already_connected():
    client = RosClient()
    client.ros.is_connected = True
    run_calls_before = list(client.ros.run_calls)

    client.ensure_connected()
    time.sleep(0.05)

    assert client.ros.run_calls == run_calls_before


def test_ensure_connected_reconnects_when_disconnected():
    client = RosClient()
    client.ros.is_connected = False

    client.ensure_connected()
    time.sleep(0.2)

    assert len(client.ros.run_calls) >= 1


# ---- ensure_connected(): 연결과 구독 준비 상태 분리 (핵심 버그 재현/수정 검증) ----

def test_ensure_connected_resubscribes_when_connected_but_not_ready():
    # ROS 연결은 True인데 아직 한 번도 구독하지 않은(subscriptions_ready=False) 상태.
    client = RosClient()
    client.ros.is_connected = True
    states = []
    client.on_connection_changed = states.append

    client.ensure_connected()

    assert client._subscriptions_ready is True
    assert [t.name for t in _FakeTopic.instances].count('/task/status') == 1
    assert states == [True]  # 재구독 성공 시에만 True로 통지


def test_ensure_connected_after_error_resubscribes_while_ros_still_connected():
    # 핵심 버그 시나리오: error 이벤트로 구독은 정리됐지만 rosbridge 연결 자체는
    # 살아 있는 경우(is_connected=True), 예전 ensure_connected()는 바로 반환해
    # 구독이 영원히 복구되지 않았다.
    client = RosClient()
    client.ros.is_connected = True  # 실제 run()이 연결 성공 시 하는 것과 동일하게 표시
    client.ros.ready_callbacks[0]()  # 최초 연결 + 구독 성공
    assert client._subscriptions_ready is True

    client.ros.event_callbacks['error']()  # 구독만 정리됨
    assert client._subscriptions_ready is False
    assert client.ros.is_connected is True  # 연결 자체는 살아있음

    states = []
    client.on_connection_changed = states.append
    client.ensure_connected()  # QTimer가 호출하는 것과 동일

    assert client._subscriptions_ready is True
    assert states == [True]
    status_topics = [t for t in _FakeTopic.instances if t.name == '/task/status']
    assert len(status_topics) == 2  # 최초 1개 + 재구독 1개, 중복 없음


def test_ensure_connected_retries_after_resubscribe_failure():
    client = RosClient()
    client.ros.is_connected = True
    _FakeTopic.fail_subscribe_for = '/robot/fault'
    states = []
    client.on_connection_changed = states.append

    client.ensure_connected()  # 첫 시도는 실패

    assert client._subscriptions_ready is False
    assert states == []  # 실패 시에는 통지하지 않는다 (자동 재시도만 예약됨)

    _FakeTopic.fail_subscribe_for = None
    client.ensure_connected()  # 다음 타이머 tick에서 재시도 - 이번엔 성공

    assert client._subscriptions_ready is True
    assert states == [True]


def test_ensure_connected_does_not_duplicate_when_already_ready():
    client = RosClient()
    client.ros.is_connected = True
    client.ros.ready_callbacks[0]()  # 연결 + 구독 모두 준비 완료
    topic_count_before = len(_FakeTopic.instances)
    states = []
    client.on_connection_changed = states.append

    client.ensure_connected()  # 이미 준비된 상태 - 아무 것도 하지 않아야 한다

    assert len(_FakeTopic.instances) == topic_count_before
    assert states == []


def test_ensure_connected_does_nothing_after_close():
    client = RosClient()
    client.ros.ready_callbacks[0]()
    client.close()
    run_calls_before = list(client.ros.run_calls)
    topic_count_before = len(_FakeTopic.instances)

    client.ensure_connected()

    assert client.ros.run_calls == run_calls_before
    assert len(_FakeTopic.instances) == topic_count_before
    assert client._subscriptions_ready is False


def test_close_terminates_ros():
    client = RosClient()

    client.close()

    assert client.ros.terminate_called is True
