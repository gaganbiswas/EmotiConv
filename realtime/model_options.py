LLM_MODELS = [
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o-mini",
]

TTS_MODELS = [
    "cartesia/sonic-3",
    "cartesia/sonic-2",
    "elevenlabs/eleven_flash_v2_5",
]

STT_MODELS = [
    "deepgram/nova-3",
    "assemblyai/universal-streaming",
    "cartesia/ink-whisper",
    "whisper-local",
]


def validate_models(config: dict) -> dict:
    values = {
        "llm_model": config.get("llm_model", LLM_MODELS[0]),
        "tts_model": config.get("tts_model", "cartesia/sonic-2"),
        "stt_model": config.get("stt_model", "whisper-local"),
    }
    if values["llm_model"] not in LLM_MODELS:
        raise ValueError("Unsupported LLM model")
    if values["tts_model"] not in TTS_MODELS:
        raise ValueError("Unsupported TTS model")
    if values["stt_model"] not in STT_MODELS:
        raise ValueError("Unsupported STT model")
    return values