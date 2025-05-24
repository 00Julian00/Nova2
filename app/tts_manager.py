"""
Description: This script handles text-to-speech.
"""

from Nova2.app.inference_engines import InferenceEngineBaseTTS
from Nova2.app.tts_data import TTSConditioning
from Nova2.app.audio_manager import AudioData

class TTSManager:
    def __init__(self):
        """
        This class runs TTS inference.
        """
        self._inference_engine = None
        self._conditioning = None

        self._inference_engine_dirty = None
        self._conditioning_dirty = None

    def configure(self, inference_engine: InferenceEngineBaseTTS, conditioning: TTSConditioning) -> None:
        """
        Configure the TTS system.
        """
        if inference_engine._type != "TTS":
            raise TypeError("Inference engine must be of type \"TTS\"")

        self._inference_engine_dirty = inference_engine

        self._conditioning_dirty = conditioning

    def apply_config(self) -> None:
        """
        Applies the configuration and loads the model into memory.
        """
        if self._inference_engine_dirty is None:
            raise Exception("Failed to initialize TTS. No inference engine provided.")
        
        if self._conditioning_dirty is None:
            raise Exception("Failed to initialize TTS. No TTS conditioning provided.")

        self._inference_engine = self._inference_engine_dirty
        self._conditioning = self._conditioning_dirty

        self._inference_engine.initialize_model(self._conditioning.model) # type: ignore

    def run_inference(self, text: str) -> AudioData:
        """
        Converts text to speech and returns the audio data.

        Arguments:
            text (str): The text that will be converted to speech.

        Returns:
            AudioData: The generated audio data.
        """
        assert self._inference_engine is not None
        assert self._conditioning is not None

        split_text = text.split(". ")

        data_chunks = []

        for sentence in split_text:
            data_chunks.extend(self._inference_engine.run_inference(
                                                            conditioning=self._conditioning,
                                                            text=sentence,
                                                            stream=False
                                                            ))
        
        data = AudioData()

        data._store_audio(data_chunks)

        return data