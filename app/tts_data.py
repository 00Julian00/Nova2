"""
Description: Holds all data required to run TTS inference.
"""

class TTSConditioning:
    def __init__(
                self,
                model: str,
                voice: str,
                expressivness: float,
                stability: float,
                **kwargs
                ) -> None:
        """
        Stores all values required for TTS conditioning.
        Note that some parameters are inference engine exclusive and will be ignored if an incompatibe engine is used.
        """
        self.model = model
        self.voice = voice
        self.expressivness = expressivness
        self.stability = stability
        self.kwargs = kwargs