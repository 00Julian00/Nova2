"""
Description: Holds all data required to run the transcriptor.
"""

from typing import Optional
from dataclasses import dataclass

import torch

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
    microphone_index: int = 0
    device: str = "cuda"
    voice_boost: float = 10.0
    vad_threshold: float = 0.95
    voice_similarity_threshold: float = 0.8