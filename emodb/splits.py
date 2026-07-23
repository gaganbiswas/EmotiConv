from __future__ import annotations

TEST_SPK = {"13", "14"}
VAL_SPK = {"15", "16"}


def split_for_speaker(spk: str) -> str:
    if spk in TEST_SPK:
        return "test"
    if spk in VAL_SPK:
        return "val"
    return "train"


def speaker_of(uttr_id: str) -> str:
    return str(uttr_id)[:2]


EMO7_NAMES = ["neutral", "happy", "angry", "sad", "fear", "disgust", "boredom"]
EMO7_LABEL = {name: i for i, name in enumerate(EMO7_NAMES)}

RAW_TO_EMO7 = {
    "neutral": "neutral",
    "happiness": "happy",
    "anger": "angry",
    "sadness": "sad",
    "fear": "fear",
    "disgust": "disgust",
    "boredom": "boredom",
}

EMO4_NAMES = ["neutral", "happy", "angry", "sad"]
EMO4_LABEL = {name: i for i, name in enumerate(EMO4_NAMES)}


def names_for(n_classes: int) -> list[str]:
    if n_classes == 4:
        return EMO4_NAMES
    if n_classes == 7:
        return EMO7_NAMES
    raise ValueError(f"Unsupported class count: {n_classes}")


def label_map_for(n_classes: int) -> dict[str, int]:
    return EMO4_LABEL if n_classes == 4 else EMO7_LABEL
