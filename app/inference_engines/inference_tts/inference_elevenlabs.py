from typing import Literal

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

from .inference_base_tts import InferenceEngineBaseTTS
from ... import security_manager
from ...security_data import Secrets
from ...tts_data import TTSConditioning

class InferenceEngineElevenlabs(InferenceEngineBaseTTS):
    def __init__(self) -> None:
        """
        This class provides the interface to run inference via the elevenlabs API.
        """
        self._key_manager = security_manager.SecretsManager()

        self._model = None

        super().__init__()

        self.is_local = False

    def initialize_model(self, model: Literal["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5"]) -> None:
        key = self._key_manager.get_secret(Secrets.ELEVENLABS_API)

        if not key:
            raise ValueError("Elevenlabs API key not found")
        
        self._elevenlabs_client = ElevenLabs(
            api_key=key
        )

        self._model = model

    def is_model_ready(self) -> bool:
        return self._model != None
    
    def get_current_model(self) -> str:
        return self._model
    
    def free(self) -> None:
        self._model = None

    def run_inference(self, text: str, conditioning: TTSConditioning, stream: bool) -> bytes:
        voice = Voice(
                    voice_id=conditioning.voice,
                    settings=VoiceSettings(
                                        stability=conditioning.stability,
                                        similarity_boost=conditioning.similarity_boost,
                                        style=conditioning.expressivness,
                                        use_speaker_boost=conditioning.use_speaker_boost
                                        )
                    )
        
        return list(self._elevenlabs_client.generate(
            text = text,
            voice=voice,
            model = self._model,
            stream = False
        ))