"""
Description: Holds all data required to run the transcriptor.
"""

from typing import Literal, Optional
from dataclasses import dataclass

import torch

from .shared_types import (
    WordBase,
    TranscriptorConditioningBase
)

@dataclass
class Word(WordBase):
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    speaker_embedding: Optional[torch.FloatTensor] = None

@dataclass
class TranscriptorConditioning(TranscriptorConditioningBase):
    microphone_index: int = 0
    model: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2"] = "deepdml/faster-whisper-large-v3-turbo-ct2"
    device: Literal["cpu", "cuda"] = "cuda"
    voice_boost: float = 10.0
    language: Optional[str] = None
    vad_threshold: float = 0.95
    voice_similarity_threshold: float = 0.8