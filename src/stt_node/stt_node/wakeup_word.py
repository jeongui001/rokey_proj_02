import math

import numpy as np
from scipy.signal import resample

# 마이크 음량 게이지용 dB 범위 - 원점(0dB)은 int16 풀스케일, floor는 "이 밑은
# 사실상 무음"으로 간주하는 하한선이다. 실측 캘리브레이션 값이 아니라(마이크/거리
# 마다 절대 음량은 다르게 마련) "지금 소리가 들어오고 있는지"만 상대적으로 보여주는
# 표준 VU 미터식 스케일이다.
_LEVEL_FLOOR_DB = -60.0


def _rms_to_level(rms: float) -> float:
    """RMS(int16 진폭)를 0.0~1.0 정규화 음량으로 변환한다(dB 스케일)."""
    if rms <= 0.0:
        return 0.0
    db = 20.0 * math.log10(rms / 32768.0)
    return min(max((db - _LEVEL_FLOOR_DB) / -_LEVEL_FLOOR_DB, 0.0), 1.0)


class WakeupWord:
    """openwakeword 기반 로컬/무료 웨이크워드 감지.

    유료 OpenAI STT API를 호출하기 전에 이 단계를 거쳐, 웨이크워드가 확인된
    발화에 대해서만 API를 호출한다(비용 절감이 목적이라 이 감지 자체는 반드시
    네트워크 호출 없이 로컬에서 끝나야 한다). 모델 로딩(openwakeword.utils.
    download_models + Model 생성)은 첫 is_wakeup() 호출까지 지연시킨다 - 생성자
    시점에 무거운 로딩/네트워크 접근이 일어나지 않게 하기 위함(테스트 용이성).

    model_path는 실제 파일 경로여야 한다(openwakeword.Model이 존재 여부를
    os.path.exists로 확인함 - 패키지 이름만으로는 못 찾는다).
    """

    def __init__(self, model_path: str, mic_rate: int, buffer_size: int, threshold: float = 0.3):
        self.model_path = model_path
        self.model_name = model_path.rsplit('/', 1)[-1].split('.', 1)[0]
        self.mic_rate = mic_rate
        self.buffer_size = buffer_size
        self.threshold = threshold
        self._model = None  # 지연 로딩, _ensure_model 참고
        self.last_level = 0.0  # 가장 최근 is_wakeup() 호출의 정규화 마이크 음량(0.0~1.0)

    def _ensure_model(self):
        if self._model is None:
            import openwakeword
            from openwakeword.model import Model
            openwakeword.utils.download_models()  # 이미 캐시돼 있으면 네트워크 호출 없음
            self._model = Model(wakeword_models=[self.model_path])
        return self._model

    def is_wakeup(self, stream) -> bool:
        """stream(PyAudio 스트림)에서 buffer_size 샘플만큼 읽어 웨이크워드 신뢰도를
        확인한다. threshold를 넘으면 True."""
        model = self._ensure_model()
        audio_chunk = np.frombuffer(
            stream.read(self.buffer_size, exception_on_overflow=False), dtype=np.int16)
        self.last_level = _rms_to_level(
            float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))))
        if self.mic_rate != 16000:
            audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / self.mic_rate))
        outputs = model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        return confidence > self.threshold


__all__ = ['WakeupWord']
