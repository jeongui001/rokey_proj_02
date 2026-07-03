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
