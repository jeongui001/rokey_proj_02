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
