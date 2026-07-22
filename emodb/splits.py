"""Shared label schemes and speaker-independent split for the in-corpus EmoDB experiments.

EmoDB has 10 speakers (03, 08-16) with an almost perfectly balanced number of
utterances per speaker (~80 each). We hold out two speakers for validation and two
for test, giving a strict *speaker-independent* evaluation that mirrors the
session-independent protocol used for IEMOCAP.
"""
from __future__ import annotations

# Speaker-independent split (speaker id = first two characters of the utterance id).
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


# 7-class scheme (all of EmoDB). Order chosen so the four IEMOCAP-overlapping
# categories come first and keep the same relative order as the 4-class scheme.
EMO7_NAMES = ["neutral", "happy", "angry", "sad", "fear", "disgust", "boredom"]
EMO7_LABEL = {name: i for i, name in enumerate(EMO7_NAMES)}

# Raw EmoDB emotion string -> 7-class short name.
RAW_TO_EMO7 = {
    "neutral": "neutral",
    "happiness": "happy",
    "anger": "angry",
    "sadness": "sad",
    "fear": "fear",
    "disgust": "disgust",
    "boredom": "boredom",
}

# 4-class scheme (the IEMOCAP-overlapping subset). Same order and label ids as
# iemocap/confusion_matrix.CLASS_NAMES[4] so results line up directly with the
# cross-corpus (IEMOCAP->EmoDB) confusion matrix.
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
