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
