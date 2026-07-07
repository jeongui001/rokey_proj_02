import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SttNode(Node):
    """마이크 -> VAD 발화구간 검출 -> 로컬 Whisper -> 텍스트(전체 계획.md 1.2절).

    핵심 로직 3개(_read_audio_chunk/_detect_voice_activity/_run_whisper)가
    전부 NotImplementedError 스텁이라, 지금은 사실상 뼈대(스레드 + 버퍼링 구조)만
    있는 상태다. 실제로 값이 흐르게 하려면 이 3개부터 채워야 한다.
    """

    def __init__(self):
        super().__init__('stt_node')
        self.pub_command = self.create_publisher(String, '/user_command/text', 10)  # 서브스크라이버: task_manager
        self._stop_event = threading.Event()
        # 오디오 캡처는 블로킹 I/O라 rclpy.spin()과 같은 스레드에서 돌리면 콜백이 막힌다.
        # 그래서 별도 스레드에서 무한 루프로 돌리고, destroy_node()에서 stop_event로 정지시킨다.
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
        """오디오 청크를 계속 읽으면서 발화 구간을 하나의 버퍼로 모으는 루프.
        말하는 동안(is_speech=True)은 버퍼에 계속 쌓고, 말이 멈추면(is_speech=False)
        그때까지 모은 버퍼 전체를 한 번에 Whisper에 넘긴다 - 즉 "문장 단위"로
        인식하지 프레임 단위로 인식하지 않는다."""
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
                # 말도 없고 모아둔 버퍼도 없으면 그냥 계속 대기
                continue
            # 말이 멈춘 시점 - 지금까지 모은 발화 전체를 한 번에 인식
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
