import io
import json
import os
import tempfile
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
import scipy.io.wavfile as wav_io
import sounddevice as sd
from openai import OpenAI

from stt_node.mic_controller import MicConfig, MicController
from stt_node.wakeup_word import WakeupWord

WAKEWORD_MODEL_FILENAME = 'hello_rokey_8332_32.tflite'


class SttNode(Node):
    """마이크 -> 웨이크워드 감지(openwakeword, 로컬/무료) -> 고정 구간 녹음 ->
    OpenAI STT(whisper-1) -> 텍스트(전체 계획.md 1.2절).

    웨이크워드("헬로 로키" - resource/hello_rokey_8332_32.tflite, cobot_ws/src/
    cobot2_ws/voice_processing 재사용)가 확인된 뒤에만 유료 OpenAI API를 호출해
    상시 녹음을 전부 API로 보내는 것보다 비용을 줄인다. 다른 문구로 바꾸려면
    새 tflite 모델을 별도로 학습해야 한다(이 파일 수정만으로는 안 됨).

    start()를 호출하기 전까지는 마이크/모델에 전혀 접근하지 않는다 - 생성자에서
    바로 시작하면 단위 테스트마다 실제 오디오 장치를 건드리게 되므로, main()에서
    명시적으로 start()를 부르는 시점부터 백그라운드 캡처 스레드가 돈다.
    """

    def __init__(self):
        super().__init__('stt_node')
        self.pub_command = self.create_publisher(String, '/user_command/text', 10)  # 서브스크라이버: task_manager
        # 서브스크라이버: operator_gui(마이크 음량 게이지) - 웨이크워드 대기 중에만
        # 갱신된다(명령 녹음 5초 구간은 sounddevice 블로킹 호출이라 청크 단위 갱신이 없음).
        self.pub_mic_level = self.create_publisher(Float32, '/stt/mic_level', 10)
        self.pub_status = self.create_publisher(String, '/stt/status', 10)
        self.pub_debug_events = self.create_publisher(String, '/debug/events', 10)

        self.declare_parameter('mic.device_index', -1)  # -1이면 PyAudio 기본 입력 장치
        self.declare_parameter('mic.rate', 48000)
        self.declare_parameter('mic.chunk', 12000)
        self.declare_parameter('mic.buffer_size', 24000)
        # 0.3에서는 실측상 놓치는 경우가 많아 0.2로 낮춤 - 오탐(false positive)이
        # 늘면 command.silence_rms_threshold/no_speech_prob 필터가 2차로 걸러준다.
        # 여전히 놓치면 `ros2 param set /stt_node wakeword.threshold <값>`으로는
        # 반영 안 됨(생성자에서 WakeupWord에 값을 한 번만 복사해 사용) - 이 기본값을
        # 더 낮추거나 launch 시 --ros-args -p wakeword.threshold:=<값>으로 재시작해야 함.
        self.declare_parameter('wakeword.threshold', 0.2)
        self.declare_parameter('command.record_seconds', 5.0)
        self.declare_parameter('command.sample_rate', 16000)
        # 녹음 구간이 이 RMS(진폭 제곱평균제곱근, int16 기준) 미만이면 거의 무음으로
        # 보고 OpenAI 호출 자체를 건너뛴다 - Whisper는 무음/저신호 구간을 주면 "시청해
        # 주셔서 감사합니다" 같은 문장을 지어내는(hallucination) 버릇이 있다(유튜브
        # 자막으로 학습된 부작용, 잘 알려진 현상). 마이크/환경마다 잡음 크기가 달라
        # 실측 후 조정이 필요할 수 있다.
        self.declare_parameter('command.silence_rms_threshold', 150.0)
        # RMS 게이트를 통과할 만큼 신호는 있지만(예: 팬 소음, 발소리) 실제 발화가
        # 아닌 경우까지는 RMS만으로 못 거른다 - Whisper verbose_json 응답의
        # no_speech_prob(구간이 무음/비음성일 확률)이 이 값 이상이면 환각으로 보고
        # 버린다 (_run_whisper 참고).
        self.declare_parameter('command.no_speech_prob_threshold', 0.6)
        # DEBUG_LOG: 실기 디버깅용 구조화 이벤트. 안정화 후 GUI/로그 정책 확정 시 제거 가능.
        self.declare_parameter('debug.publish_events', True)

        package_share = get_package_share_directory('stt_node')
        load_dotenv(dotenv_path=os.path.join(package_share, 'resource', '.env'))
        api_key = os.getenv('OPENAI_API_KEY')
        self._openai_client = OpenAI(api_key=api_key) if api_key else None
        if self._openai_client is None:
            self.get_logger().error(
                'OPENAI_API_KEY가 설정되지 않았습니다(resource/.env 확인 필요) - '
                '웨이크워드는 감지되지만 STT 호출은 계속 실패합니다.')

        device_index = int(self.get_parameter('mic.device_index').value)
        self._mic = MicController(MicConfig(
            chunk=int(self.get_parameter('mic.chunk').value),
            rate=int(self.get_parameter('mic.rate').value),
            buffer_size=int(self.get_parameter('mic.buffer_size').value),
            device_index=None if device_index < 0 else device_index,
        ))
        self._wakeup_word = WakeupWord(
            os.path.join(package_share, 'resource', WAKEWORD_MODEL_FILENAME),
            mic_rate=self._mic.config.rate,
            buffer_size=self._mic.config.buffer_size,
            threshold=float(self.get_parameter('wakeword.threshold').value))

        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

    def _checkpoint_event(
            self, phase, checkpoint_id, status, message, data=None,
            *, throttle_s=None, log=False):
        """파이프라인 점검.md의 Phase 체크리스트에 대응하는 이벤트를 발행한다."""
        now = time.monotonic()
        key = (checkpoint_id, status)
        if throttle_s is not None:
            last = getattr(self, '_checkpoint_event_last', {}).get(key, 0.0)
            if now - last < throttle_s:
                return
            if not hasattr(self, '_checkpoint_event_last'):
                self._checkpoint_event_last = {}
            self._checkpoint_event_last[key] = now
        payload = {
            'phase': phase,
            'checkpoint_id': checkpoint_id,
            'status': status,
            'message': message,
            'data': data or {},
            'node': self.get_name(),
            'stamp_monotonic': now,
        }
        if bool(self.get_parameter('debug.publish_events').value):
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub_debug_events.publish(msg)
        if log:
            text = f'[CHECKPOINT][{phase}/{checkpoint_id}] status={status} message={message}'
            if status == 'FAIL':
                self.get_logger().error(text)
            else:
                self.get_logger().info(text)

    def _publish_status(self, state, detail='', data=None):
        msg = String()
        msg.data = json.dumps({
            'state': state,
            'detail': detail,
            'data': data or {},
        }, ensure_ascii=False)
        self.pub_status.publish(msg)

    def start(self):
        """백그라운드 오디오 캡처를 시작한다. main()에서만 호출 - 테스트는 이걸
        호출하지 않아 실제 마이크/모델에 접근하지 않는다."""
        self._capture_thread.start()

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            self._publish_status('error', str(exc))
            return default

    def _capture_loop(self):
        # 오디오 캡처는 블로킹 I/O라 rclpy.spin()과 같은 스레드에서 돌리면 콜백이
        # 막힌다. 그래서 별도 스레드에서 무한 루프로 돌리고, destroy_node()에서
        # stop_event로 정지시킨다. 마이크/네트워크 예외로 이 스레드 자체가 죽어
        # 음성 명령이 영구히 먹통되지 않도록 매 사이클을 예외로부터 보호한다.
        while not self._stop_event.is_set():
            try:
                self._run_one_listen_cycle()
            except Exception as exc:
                self.get_logger().error(f'stt 캡처 루프 예외: {exc}')
                self._publish_status('error', f'음성 인식 오류: {exc}')
                self._stop_event.wait(1.0)

    def _run_one_listen_cycle(self):
        if not self._safe_call(self._open_mic_stream, default=False):
            self._stop_event.wait(1.0)
            return
        try:
            detected = self._wait_for_wake_word()
        finally:
            self._close_mic_stream()
        if not detected:
            return  # stop_event로 중단됨
        self.get_logger().info('웨이크워드 감지 - 명령 녹음을 시작합니다.')
        self._publish_status('wakeword_detected', 'wakeWord가 인식되었습니다. 명령어를 말해주세요.')
        audio = self._safe_call(self._record_command_audio, default=None)
        if audio is None:
            return
        self._publish_status('transcribing', '명령어를 인식하는 중입니다.')
        text = self._safe_call(self._run_whisper, audio, default=None)
        if text:
            self._on_utterance_ready(text)

    def _wait_for_wake_word(self) -> bool:
        """웨이크워드가 감지되면 True, stop_event로 중단되면 False."""
        while not self._stop_event.is_set():
            if self._safe_call(self._detect_wake_word, default=False):
                return True
        return False

    def _on_utterance_ready(self, text: str):
        msg = String()
        msg.data = text
        self.pub_command.publish(msg)
        self._publish_status('utterance_ready', text)
        self._checkpoint_event(
            'A', 'stt_utterance_published', 'PASS',
            'STT 결과 텍스트를 발행했습니다.', {'text': text})

    # ---- 오디오 I/O (PyAudio, 웨이크워드 감지용 연속 스트림) ----

    def _open_mic_stream(self) -> bool:
        self._mic.open_stream()
        return True

    def _close_mic_stream(self) -> None:
        self._mic.close_stream()

    # ---- 웨이크워드 (openwakeword, 로컬/무료) ----

    def _detect_wake_word(self) -> bool:
        detected = self._wakeup_word.is_wakeup(self._mic.stream)
        self.pub_mic_level.publish(Float32(data=self._wakeup_word.last_level))
        return detected

    # ---- 명령 녹음 + OpenAI STT ----

    def _record_command_audio(self):
        """웨이크워드 감지 직후 고정 구간(기본 5초)을 녹음해 WAV bytes로 반환한다.
        거의 무음이면 None을 반환해 OpenAI 호출 자체를 건너뛴다(_is_silent 참고).

        PyAudio 스트림은 이미 닫힌 뒤라(_run_one_listen_cycle) 여기서는 sounddevice로
        새로 녹음한다 - 검증된 기존 방식(cobot2_ws voice_processing/stt.py) 그대로."""
        duration_s = float(self.get_parameter('command.record_seconds').value)
        sample_rate = int(self.get_parameter('command.sample_rate').value)
        self._publish_status(
            'recording_command', '명령어 녹음 중입니다.',
            {'duration_s': duration_s, 'sample_rate': sample_rate})
        audio = sd.rec(
            int(duration_s * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        if self._is_silent(audio):
            rms = self._command_rms(audio)
            threshold = float(self.get_parameter('command.silence_rms_threshold').value)
            self.get_logger().info(
                f'녹음 구간이 거의 무음(rms={rms}, threshold={threshold}) - STT 호출을 건너뜁니다.')
            self._publish_status(
                'silent_skipped', '녹음 구간이 거의 무음이라 인식을 건너뜁니다.',
                {'rms': rms})
            return None
        buffer = io.BytesIO()
        buffer.name = 'command.wav'  # OpenAI SDK가 파일명 확장자로 포맷을 추론한다
        wav_io.write(buffer, sample_rate, audio)
        return buffer.getvalue()

    def _is_silent(self, audio: np.ndarray) -> bool:
        threshold = float(self.get_parameter('command.silence_rms_threshold').value)
        rms = self._command_rms(audio)
        return rms < threshold

    @staticmethod
    def _command_rms(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    # RMS 게이트를 통과하고도 Whisper가 지어내는 것으로 잘 알려진 한국어 환각
    # 문구들(유튜브 자막 학습 부작용) - no_speech_prob 필터의 2차 안전망.
    _HALLUCINATION_PHRASES = (
        '시청해주셔서 감사합니다',
        '구독과 좋아요 부탁드립니다',
        '다음 영상에서 만나요',
        '이 영상은 유료광고를 포함하고 있습니다',
    )

    def _run_whisper(self, utterance_audio: bytes) -> str:
        """utterance_audio(WAV bytes)를 OpenAI whisper-1 API로 전사한다.

        language='ko'로 고정해 무음/잡음에서 엉뚱한 언어(태국어 등)로 잘못 감지되는
        현상을 막는다. OPENAI_API_KEY가 없으면 NotImplementedError로 fail-closed -
        잘못된 키 없이 조용히 넘어가지 않고 _safe_call이 로그로 남기게 한다.

        response_format='verbose_json'으로 구간별 no_speech_prob을 받아, RMS
        게이트는 통과했지만(팬 소음 등) 실제 발화가 아닌 구간에서 Whisper가 지어낸
        텍스트를 걸러낸다(_looks_like_hallucination 참고)."""
        if self._openai_client is None:
            raise NotImplementedError('OPENAI_API_KEY 미설정 - resource/.env를 확인하세요.')
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav:
            temp_wav.write(utterance_audio)
            temp_wav.flush()
            with open(temp_wav.name, 'rb') as f:
                transcript = self._openai_client.audio.transcriptions.create(
                    model='whisper-1', file=f, language='ko',
                    response_format='verbose_json')
        text = transcript.text.strip()
        if self._looks_like_hallucination(text, getattr(transcript, 'segments', None)):
            self.get_logger().info(f'Whisper 결과를 환각으로 판단해 버립니다: "{text}"')
            self._publish_status('hallucination_dropped', '인식된 문장이 신뢰도가 낮아 버렸습니다.', {'text': text})
            return ''
        return text

    def _looks_like_hallucination(self, text: str, segments) -> bool:
        normalized = text.replace(' ', '')
        if any(phrase.replace(' ', '') in normalized for phrase in self._HALLUCINATION_PHRASES):
            return True
        if not segments:
            return False
        threshold = float(self.get_parameter('command.no_speech_prob_threshold').value)
        no_speech_probs = [
            p for p in (self._segment_no_speech_prob(seg) for seg in segments)
            if p is not None
        ]
        return bool(no_speech_probs) and max(no_speech_probs) >= threshold

    @staticmethod
    def _segment_no_speech_prob(segment):
        if isinstance(segment, dict):
            return segment.get('no_speech_prob')
        return getattr(segment, 'no_speech_prob', None)


def main(args=None):
    rclpy.init(args=args)
    node = SttNode()
    node.start()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
