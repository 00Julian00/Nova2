"""
Description: Holds all data required to run TTS inference.
"""
import torch

class TTSConditioning:
    def __init__(
                self,
                model: str = "Zyphra/Zonos-v0.1-transformer",
                voice: str = "Laura",
                language: str = "en-us",
                emotion: torch.FloatTensor = None,
                speaking_rate: int = 15,
                expressivness: float = 100,
                stability: float = 2,
                similarity_boost: float = 1.0,
                use_speaker_boost: bool = True
                ) -> None:
        """
        Stores all values required for TTS conditioning.
        Note that some parameters are inference engine exclusive and will be ignored if an incompatibe engine is used.
        """
        self.model = model
        self.voice = voice
        self.language = language
        self.emotion = emotion
        self.speaking_rate = speaking_rate
        self.expressivness = expressivness
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.use_speaker_boost = use_speaker_boost