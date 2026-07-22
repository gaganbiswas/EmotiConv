from faster_whisper import WhisperModel

class Translator:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path, language="de", task="translate"):
        segments, _ = self.model.transcribe(audio_path, language=language, task=task)
        return " ".join(seg.text.strip() for seg in segments).strip()
