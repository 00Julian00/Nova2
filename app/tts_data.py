"""
Description: Holds all data required to run TTS inference.
"""
from Nova2.app.interfaces import (
    TTSConditioningBase
)

class TTSConditioning(TTSConditioningBase):
    def __init__(
                self,
                model: str,
                inference_engine:str,
                voice: str,
                expressivness: float,
                stability: float,
                **kwargs
                ) -> None:
        self.model = model
        self.inference_engine = inference_engine
        self.voice = voice
        self.expressivness = expressivness
        self.stability = stability
        self.kwargs = kwargs