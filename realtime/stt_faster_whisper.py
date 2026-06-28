from __future__ import annotations

import numpy as np
import librosa
from livekit import rtc
from livekit.agents import stt, APIConnectOptions
from faster_whisper import WhisperModel

def _buffer_to_16k(buffer):
    frame = rtc.combine_audio_frames(buffer) if isinstance(buffer, (list, tuple)) else buffer
    data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
    if frame.num_channels > 1:
        data = data.reshape(-1, frame.num_channels).mean(axis=1)
    if frame.sample_rate != 16000:
        data = librosa.resample(data, orig_sr=frame.sample_rate, target_sr=16000)
    return data.astype(np.float32)

class FasterWhisperSTT(stt.STT):
    def __init__(self, model="small.en", device="auto", compute_type="int8", language="en"):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None, **kwargs):
        audio = _buffer_to_16k(buffer)
        text = ""
        if audio.size:
            segments, _ = self._model.transcribe(audio, language=language or self._language, beam_size=1)
            text = " ".join(s.text for s in segments).strip()
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=language or self._language, text=text)],
        )
