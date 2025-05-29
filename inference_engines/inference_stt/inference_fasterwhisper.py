from os import cpu_count
from warnings import warn

from faster_whisper import WhisperModel
from numpy import ndarray
from torch import Tensor

from Nova2.app.interfaces import STTConditioningBase, STTInferenceEngineBase, WordBase
from Nova2.app.stt_data import STTConditioning, Word

class InferenceEngineFasterWhisper(STTInferenceEngineBase):
    def __init__(self) -> None:
        """
        This class runs STT inference via faster-whisper.
        """
        self._model: WhisperModel = None # type: ignore
        self._conditioning: STTConditioning = None # type: ignore

    def initialize_model(self, conditioning: STTConditioningBase) -> None:
        self.free()
        
        self._conditioning = conditioning # type: ignore

        cpu_cores = cpu_count()

        if not cpu_cores:
            cpu_cores = 1
            warn("Failed to detect CPU core count. Defaulting to 1.")
        
        self._model = WhisperModel(
            model_size_or_path=self._conditioning.model,
            device=self._conditioning.device,
            compute_type="float32",
            cpu_threads=cpu_cores
            )
        
    def run_inference(self, audio_data: ndarray | Tensor) -> list[WordBase]:
        if type(audio_data) == Tensor:
            audio_data = audio_data.cpu().numpy()

        if self._conditioning.language != "":
            segments, info = self._model.transcribe(audio_data, beam_size=5, language=self._conditioning.language, condition_on_previous_text=False, word_timestamps=True) # type: ignore
        else:
            segments, info = self._model.transcribe(audio_data, beam_size=5, condition_on_previous_text=False, word_timestamps=True) # type: ignore | Leave the language undefined so whisper autodetects it
        transcription = []
        for segment in segments:
            for word in segment.words: # type: ignore
                transcription.append(Word(text=word.word, start=word.start, end=word.end))
        return transcription
    
    def free(self) -> None:
        del self._model

    @property
    def model(self) -> str:
        if not self._conditioning:
            return ""
        return self._conditioning.model