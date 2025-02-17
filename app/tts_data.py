"""
Description: Holds all data required to run TTS inference.
"""
import torch

class TTSConditioning:
    def __init__(
                self,
                voice: str,
                language: str = "en",
                emotion: torch.FloatTensor = None,
                speaking_rate: int = 15,
                expressivness: float = 0.5,
                stability: float = 0.5,
                similarity_boost: float = 1.0,
                use_speaker_boost: bool = True
                ) -> None:
        """
        Stores all values required for TTS conditioning.
        Note that some parameters are inference engine exclusive and will be ignored if an incompatibe engine is used.
        """
        self.voice = voice
        self.language = language
        self.emotion = emotion
        self.speaking_rate = speaking_rate
        self.expressivness = expressivness
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.use_speaker_boost = use_speaker_boost