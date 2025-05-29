"""
Description: Holds all data required to run the transcriptor.
"""
from typing import Optional
from dataclasses import dataclass

import torch
import sounddevice as sd

from Nova2.app.interfaces import (
    WordBase,
    STTConditioningBase
)

@dataclass
class Word(WordBase):
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    speaker_embedding: Optional[torch.Tensor] = None

@dataclass
class STTConditioning(STTConditioningBase):
    model: str
    inference_engine: str
    microphone_index: int = -1
    device: str = "cuda"
    voice_boost: float = 10.0
    vad_threshold: float = 0.95
    voice_similarity_threshold: float = 0.8

    def __post_init__(self):
        if self.microphone_index == -1: # No microphone selected
            self.microphone_index = sd.default.device[0] # type: ignore