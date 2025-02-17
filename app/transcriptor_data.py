"""
Description: Holds all data required to run the transcriptor.
"""
from typing import Literal

import torch

class Word:
    def __init__(
            self,
            text: str = "",
            start: float = 0,
            end: float = 0,
            speaker_embedding: torch.FloatTensor = None
            ) -> None:
        """
        A class to represent a word in a transcription.

        This class holds a singular word from a transcription, together with other relevant information.

        Arguments:
            text (str, optional): The text of the word. Defaults to "".
            start (float, optional): The start time of the word in seconds. Defaults to 0.
            end (float, optional): The end time of the word in seconds. Defaults to 0.
            speaker_embedding (torch.FloatTensor, optional): The speaker embedding of the voice that said the word. Defaults to None.
        """
        self.text = text
        self.start = start
        self.end = end
        self.speaker_embedding = speaker_embedding

class TranscriptorConditioning:
    def __init__(
                self,
                microphone_index: int = 0,
                model: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2"] = "large-v3",
                device: Literal["cpu", "cuda"] = "cuda",
                voice_boost: float = 10.0,
                language: str = None
                ) -> None:
        """
        Stores all values required for transcriptor conditioning.

        Arguments:
            microphone_index (int): The index of the microphone to use for recording.
            model (Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2"]): The Whisper model to use. Defaults to "large-v3".
            device (Literal["cpu", "cuda"]): The device to use for the computations. Defaults to "cuda" or "cpu" if cuda is not available.
            voice_boost (float): How much to boost the voice in the audio preprocessing stage. Setting it to 0 disables this feature. Defaults to 10.0.
            language (str): The language to use for the transcription. Must be a valid language code. If None, the language will be autodetected. If possible, the language should be set to improve accuracy. Defaults to None.
        """
        self.microphone_index = microphone_index
        self.model = model
        self.device = device
        self.voice_boost = voice_boost
        self.language = language