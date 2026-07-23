import asyncio
import os
from pathlib import Path

import numpy as np
from livekit.agents import APIConnectOptions, tts, utils
from livekit.agents.tts import ChunkedStream, TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

HERE = Path(__file__).resolve().parent

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
DEFAULT_REF_AUDIO = str(HERE / "voices" / "female_ref.wav")
DEFAULT_REF_TEXT = "Hi there, it's lovely to meet you. I'm really glad we get to talk today."
DEFAULT_SAMPLE_RATE = 24000

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

class _CudaBackend:

    def __init__(self, model_id, ref_audio, ref_text, language):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.ref_audio, self.ref_text, self.language = ref_audio, ref_text, language
        self.model = Qwen3TTSModel.from_pretrained(
            model_id, device_map="cuda:0", dtype=torch.bfloat16)
        self.sample_rate = DEFAULT_SAMPLE_RATE

    def infer(self, text: str) -> np.ndarray:
        wavs, sr = self.model.generate_voice_clone(
            text=text, language=self.language,
            ref_audio=self.ref_audio, ref_text=self.ref_text)
        self.sample_rate = int(sr)
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1)

class _MlxBackend:

    def __init__(self, model_id, ref_audio, ref_text, language):
        from mlx_audio.tts.utils import load_model

        self.ref_audio, self.ref_text, self.language = ref_audio, ref_text, language
        self.model = load_model(model_id)
        self.sample_rate = DEFAULT_SAMPLE_RATE

    def infer(self, text: str) -> np.ndarray:
        results = list(self.model.generate(
            text=text, ref_audio=self.ref_audio, ref_text=self.ref_text))
        r = results[0]
        self.sample_rate = int(getattr(r, "sample_rate", self.sample_rate))
        return np.asarray(r.audio, dtype=np.float32).reshape(-1)

class QwenTTS(tts.TTS):
    def __init__(self, *, model=None, mlx_model=None, ref_audio=None,
                 ref_text=None, language=None):
        model = model or os.getenv("QWEN_TTS_MODEL", DEFAULT_MODEL)
        mlx_model = mlx_model or os.getenv("QWEN_TTS_MLX_MODEL", DEFAULT_MLX_MODEL)
        ref_audio = ref_audio or os.getenv("QWEN_REF_AUDIO", DEFAULT_REF_AUDIO)
        ref_text = ref_text or os.getenv("QWEN_REF_TEXT", DEFAULT_REF_TEXT)
        language = language or os.getenv("QWEN_TTS_LANG", "English")

        if "://" not in str(ref_audio) and not Path(ref_audio).exists():
            raise FileNotFoundError(
                f"Qwen TTS reference audio not found: {ref_audio}\n"
                "Add a 3-10 s female voice clip there (or set QWEN_REF_AUDIO to a path/URL) "
                "and set QWEN_REF_TEXT to its exact transcript. The Base model clones this voice.")

        if _cuda_available():
            self._backend = _CudaBackend(model, ref_audio, ref_text, language)
        else:
            self._backend = _MlxBackend(mlx_model, ref_audio, ref_text, language)

        try:
            self._backend.infer("Hello.")
        except Exception:
            pass

        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=self._backend.sample_rate, num_channels=1)

    def synthesize(self, text: str, *,
                   conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> ChunkedStream:
        return _QwenStream(tts=self, input_text=text, conn_options=conn_options)

class _QwenStream(ChunkedStream):
    def __init__(self, *, tts: QwenTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: QwenTTS = tts

    async def _run(self, output_emitter) -> None:
        wav = await asyncio.to_thread(self._tts._backend.infer, self.input_text)
        pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm)
        output_emitter.flush()
