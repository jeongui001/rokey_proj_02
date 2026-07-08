import pyaudio


class MicConfig:
    """PyAudio 입력 스트림 설정 (웨이크워드 감지용 연속 스트림 전용 -
    명령 녹음은 sounddevice로 별도 처리한다, stt_node.py 참고)."""

    def __init__(
            self, chunk: int = 12000, rate: int = 48000, channels: int = 1,
            fmt: int = pyaudio.paInt16, buffer_size: int = 24000,
            device_index: int = None):
        self.chunk = chunk
        self.rate = rate
        self.channels = channels
        self.fmt = fmt
        self.buffer_size = buffer_size
        self.device_index = device_index  # None이면 PyAudio 기본 입력 장치


class MicController:
    """PyAudio 마이크 스트림 수명주기 관리자."""

    def __init__(self, config: MicConfig = None):
        self.config = config or MicConfig()
        self.audio = None
        self.stream = None

    def open_stream(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.config.fmt,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            input_device_index=self.config.device_index,
            frames_per_buffer=self.config.chunk,
        )

    def close_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None


__all__ = ['MicController', 'MicConfig']
