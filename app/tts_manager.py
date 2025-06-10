"""
Description: This script handles text-to-speech.
"""

from Nova2.app.interfaces import TTSInferenceEngineBase
from Nova2.app.tts_data import TTSConditioning
from Nova2.app.audio_manager import AudioData
from Nova2.app.inference_engine_manager import InferenceEngineManager
from Nova2.app.helpers import is_configured

class TTSManager:
    def __init__(self):
        """
        This class runs TTS inference.
        """
        self._conditioning: TTSConditioning = None # type: ignore
        self._conditioning_dirty = None

        self._inference_engine_manager = InferenceEngineManager()

    def configure(self, conditioning: TTSConditioning) -> None:
        """
        Configure the TTS system.
        """
        if not self._conditioning_dirty:
            raise Exception("Failed to initialize TTS. No TTS conditioning provided.")
        self._conditioning_dirty = conditioning

    def apply_config(self) -> None:
        """
        Applies the configuration and loads the model into memory.
        """  
        if not self._conditioning_dirty:
            raise Exception("Failed to initialize TTS. No TTS conditioning provided.")

        self._conditioning = self._conditioning_dirty

        self._inference_engine: TTSInferenceEngineBase = self._inference_engine_manager.request_engine(
            self._conditioning.inference_engine,
            "TTS"
            ) # type: ignore

        self._inference_engine.initialize_model(self._conditioning) # type: ignore
    
    @is_configured
    def run_inference(self, text: str) -> AudioData:
        """
        Converts text to speech and returns the audio data.

        Arguments:
            text (str): The text that will be converted to speech.

        Returns:
            AudioData: The generated audio data.
        """
        data = self._inference_engine.run_inference(
            conditioning=self._conditioning,
            text=text,
            stream=False
            )

        return data # type: ignore