"""
Description: Holds all data required to run TTS inference.
"""

from .shared_types import (
    TTSConditioningBase
)

class TTSConditioning(TTSConditioningBase):
    def __init__(
                self,
                model: str,
                voice: str,
                expressivness: float,
                stability: float,
                **kwargs
                ) -> None:
        self.model = model
        self.voice = voice
        self.expressivness = expressivness
        self.stability = stability
        self.kwargs = kwargs