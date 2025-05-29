import os
from typing import Literal
import warnings

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

from Nova2.app.interfaces import TTSInferenceEngineBase
from Nova2.app.security_manager import SecretsManager
from Nova2.app.tts_data import TTSConditioning
from Nova2.app.audio_data import AudioData

class InferenceEngineElevenlabs(TTSInferenceEngineBase):
    def __init__(self) -> None:
        """
        This class provides the interface to run inference via the elevenlabs API.
        """
        self._key_manager = SecretsManager()

        self._model: str = ""

        super().__init__()

        self.is_local = False

    def initialize_model(self, model: Literal["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5"]) -> None:
        key = os.getenv("ELEVENLABS_API_KEY")

        if not key:
            raise ValueError("Elevenlabs API key not found")
        
        self._elevenlabs_client = ElevenLabs(
            api_key=key
        )

        self._model = model

    def clone_voice(self, audio_dir: str, name: str) -> None:
        raise NotImplementedError("Cloning voices is not supported by the Elevenlabs inference engine. Please use the Elevenlabs web interface to clone voices.")

    def is_model_ready(self) -> bool:
        return self._model != None
    
    @property
    def model(self) -> str:
        return self._model
    
    def free(self) -> None:
        self._model = ""

    def run_inference(self, text: str, conditioning: TTSConditioning, stream: bool = False) -> AudioData: # type: ignore
        if stream:
            warnings.warn("Streaming is currently not supported by the Elevenlabs inference engine")
        
        voice = Voice(
                    voice_id=conditioning.voice,
                    settings=VoiceSettings(
                                        stability=conditioning.stability,
                                        similarity_boost=conditioning.kwargs["similarity_boost"],
                                        style=conditioning.expressivness,
                                        use_speaker_boost=conditioning.kwargs["use_speaker_boost"]
                                        )
                    )
        
        data = self._elevenlabs_client.generate(
            text = text,
            voice=voice,
            model = self._model,
            stream = False
        )

        return AudioData(list(data))