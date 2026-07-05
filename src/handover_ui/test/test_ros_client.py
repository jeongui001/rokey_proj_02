import base64
import time

import roslibpy
import pytest

from handover_ui.ros_client import DEFAULT_CAMERA_TOPIC, RosClient


class _FakeTopic:
    instances = []
    raise_on_unsubscribe = False
    raise_on_unadvertise = False

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
    monkeypatch.delenv('HANDOVER_UI_CAMERA_TOPIC', raising=False)
    client = RosClient()
    assert client.camera_topic == DEFAULT_CAMERA_TOPIC


def test_camera_topic_from_env_var(monkeypatch):
    monkeypatch.setenv('HANDOVER_UI_CAMERA_TOPIC', '/custom/camera/compressed')
    client = RosClient()
    assert client.camera_topic == '/custom/camera/compressed'


def test_camera_topic_constructor_arg_overrides_env(monkeypatch):
    monkeypatch.setenv('HANDOVER_UI_CAMERA_TOPIC', '/env/camera')
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


def test_close_terminates_ros():
    client = RosClient()

    client.close()

    assert client.ros.terminate_called is True
