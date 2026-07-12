import json

import numpy as np
import pytest
import rclpy
from std_msgs.msg import Float32, String

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


def test_start_not_called_means_no_hardware_touched(node):
    # 생성자만으로는 캡처 스레드가 시작되지 않는다 - start()를 명시적으로 호출해야만
    # 실제 마이크/웨이크워드 모델에 접근한다(단위 테스트가 하드웨어에 의존하지 않기 위함).
    assert node._capture_thread.is_alive() is False


def test_safe_call_swallows_not_implemented(node):
    def _stub():
        raise NotImplementedError('테스트용')

    result = node._safe_call(_stub, default='fallback')
    assert result == 'fallback'


def test_safe_call_does_not_swallow_other_exceptions(node):
    def _stub():
        raise RuntimeError('테스트용')

    with pytest.raises(RuntimeError):
        node._safe_call(_stub, default='fallback')


# ---- _run_whisper (OpenAI STT) ----

def test_run_whisper_raises_when_no_api_key(node):
    node._openai_client = None
    with pytest.raises(NotImplementedError):
        node._run_whisper(b'')


class _FakeTranscript:
    def __init__(self, text, segments=None):
        self.text = text
        self.segments = segments


class _FakeTranscriptions:
    def __init__(self, transcript):
        self._transcript = transcript
        self.calls = []

    def create(self, model, file, language=None, response_format=None):
        self.calls.append((model, language, response_format))
        return self._transcript


def _client_returning(transcript):
    transcriptions = _FakeTranscriptions(transcript)

    class _FakeAudio:
        pass

    audio = _FakeAudio()
    audio.transcriptions = transcriptions

    class _FakeClient:
        pass

    client = _FakeClient()
    client.audio = audio
    return client, transcriptions


def test_run_whisper_calls_openai_and_returns_text(node):
    client, transcriptions = _client_returning(
        _FakeTranscript('스패너 갖다줘', segments=[{'no_speech_prob': 0.05}]))
    node._openai_client = client

    result = node._run_whisper(b'RIFF....fake-wav-bytes')

    assert result == '스패너 갖다줘'
    # language='ko' 고정 - 무음/잡음에서 엉뚱한 언어로 오감지되는 걸 방지
    # (Whisper hallucination 완화, /user_command/text 진단 세션에서 발견).
    assert transcriptions.calls == [('whisper-1', 'ko', 'verbose_json')]


def test_run_whisper_drops_high_no_speech_prob_result(node):
    client, _ = _client_returning(
        _FakeTranscript('시청해주셔서 감사합니다', segments=[{'no_speech_prob': 0.92}]))
    node._openai_client = client

    assert node._run_whisper(b'RIFF....fake-wav-bytes') == ''


def test_run_whisper_drops_known_hallucination_phrase_even_without_segments(node):
    client, _ = _client_returning(
        _FakeTranscript('시청해주셔서 감사합니다', segments=None))
    node._openai_client = client

    assert node._run_whisper(b'RIFF....fake-wav-bytes') == ''


def test_run_whisper_keeps_low_no_speech_prob_result(node):
    client, _ = _client_returning(
        _FakeTranscript('펜치 갖다줘', segments=[{'no_speech_prob': 0.1}]))
    node._openai_client = client

    assert node._run_whisper(b'RIFF....fake-wav-bytes') == '펜치 갖다줘'


# ---- _detect_wake_word (openwakeword 위임) ----

def test_detect_wake_word_delegates_to_wakeup_word(node):
    calls = []

    class _FakeWakeupWord:
        last_level = 0.0

        def is_wakeup(self, stream):
            calls.append(stream)
            return True

    node._wakeup_word = _FakeWakeupWord()
    node._mic.stream = object()

    assert node._detect_wake_word() is True
    assert calls == [node._mic.stream]


def test_detect_wake_word_publishes_mic_level(node):
    class _FakeWakeupWord:
        last_level = 0.42

        def is_wakeup(self, stream):
            return False

    node._wakeup_word = _FakeWakeupWord()
    node._mic.stream = object()
    received = []
    node.create_subscription(Float32, '/stt/mic_level', lambda m: received.append(m.data), 10)

    node._detect_wake_word()
    rclpy.spin_once(node, timeout_sec=1.0)

    assert received == pytest.approx([0.42])


# ---- _record_command_audio (sounddevice -> WAV bytes, 무음 필터링) ----

def test_record_command_audio_returns_valid_wav(node, monkeypatch):
    sample_rate = int(node.get_parameter('command.sample_rate').value)
    duration_s = float(node.get_parameter('command.record_seconds').value)
    # 무음 필터(RMS 임계값)를 넘도록 충분히 큰 진폭의 신호를 준다.
    fake_audio = np.full((int(sample_rate * duration_s), 1), 5000, dtype='int16')
    rec_calls = []

    def fake_rec(frames, samplerate, channels, dtype):
        rec_calls.append((frames, samplerate, channels, dtype))
        return fake_audio

    monkeypatch.setattr('stt_node.stt_node.sd.rec', fake_rec)
    monkeypatch.setattr('stt_node.stt_node.sd.wait', lambda: None)

    wav_bytes = node._record_command_audio()

    assert wav_bytes[:4] == b'RIFF'
    assert wav_bytes[8:12] == b'WAVE'
    assert rec_calls == [(int(duration_s * sample_rate), sample_rate, 1, 'int16')]


def test_record_command_audio_returns_none_when_silent(node, monkeypatch):
    sample_rate = int(node.get_parameter('command.sample_rate').value)
    duration_s = float(node.get_parameter('command.record_seconds').value)
    silent_audio = np.zeros((int(sample_rate * duration_s), 1), dtype='int16')

    monkeypatch.setattr('stt_node.stt_node.sd.rec', lambda *a, **k: silent_audio)
    monkeypatch.setattr('stt_node.stt_node.sd.wait', lambda: None)

    assert node._record_command_audio() is None


def test_is_silent_respects_threshold_parameter(node):
    loud = np.full((100, 1), 5000, dtype='int16')
    quiet = np.full((100, 1), 10, dtype='int16')

    assert node._is_silent(loud) is False
    assert node._is_silent(quiet) is True


# ---- _run_one_listen_cycle (전체 흐름 결선) ----

def test_run_one_listen_cycle_full_flow_publishes_text(node):
    node._open_mic_stream = lambda: True
    node._close_mic_stream = lambda: None
    node._detect_wake_word = lambda: True
    node._record_command_audio = lambda: b'fake-wav'
    node._run_whisper = lambda audio: '스패너 갖다줘'

    published = []
    node.pub_command.publish = published.append

    node._run_one_listen_cycle()

    assert len(published) == 1
    assert published[0].data == '스패너 갖다줘'


def test_run_one_listen_cycle_skips_recording_when_wake_word_not_detected(node):
    node._open_mic_stream = lambda: True
    node._close_mic_stream = lambda: None
    node._stop_event.set()  # _wait_for_wake_word가 즉시 False를 반환하게 함

    def _fail_if_called():
        raise AssertionError('웨이크워드 감지 없이 녹음을 시작하면 안 됨')

    node._record_command_audio = _fail_if_called

    published = []
    node.pub_command.publish = published.append

    node._run_one_listen_cycle()

    assert published == []


def test_run_one_listen_cycle_skips_when_mic_open_fails(node):
    node._open_mic_stream = lambda: False

    def _fail_if_called():
        raise AssertionError('마이크 오픈 실패 시 웨이크워드 대기를 시작하면 안 됨')

    node._detect_wake_word = _fail_if_called
    node._stop_event.wait = lambda timeout: None  # 재시도 대기를 실제로 기다리지 않음

    node._run_one_listen_cycle()  # 예외 없이 조용히 반환되어야 함


def test_run_one_listen_cycle_closes_mic_even_if_no_text(node):
    close_calls = []
    node._open_mic_stream = lambda: True
    node._close_mic_stream = lambda: close_calls.append(True)
    node._detect_wake_word = lambda: True
    node._record_command_audio = lambda: None  # 녹음 실패 상황

    def _fail_if_called(audio):
        raise AssertionError('녹음이 없으면 STT를 호출하면 안 됨')

    node._run_whisper = _fail_if_called

    node._run_one_listen_cycle()

    assert close_calls == [True]


def test_on_utterance_ready_publishes_checkpoint(node):
    published = []
    node.pub_debug_events.publish = published.append

    node._on_utterance_ready('스패너 갖다줘')

    payload = json.loads(published[-1].data)
    assert payload['phase'] == 'A'
    assert payload['checkpoint_id'] == 'stt_utterance_published'
    assert payload['status'] == 'PASS'
    assert payload['data']['text'] == '스패너 갖다줘'
