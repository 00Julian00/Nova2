from typing import Literal
import warnings

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

from Nova2.app.inference_engines.inference_tts.inference_base_tts import InferenceEngineBaseTTS
from Nova2.app.security_manager import SecretsManager
from Nova2.app.security_data import Secrets
from Nova2.app.tts_data import TTSConditioning
from Nova2.app.helpers import deprecated

class InferenceEngineElevenlabs(InferenceEngineBaseTTS):
    def __init__(self) -> None:
        """
        This class provides the interface to run inference via the elevenlabs API.
        """
        self._key_manager = SecretsManager()

        self._model: str = ""

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
    
    @deprecated(warning=f"Function 'get_current_model' is deprecated and will be removed in a future update. Use the property 'model' instead.")
    def get_current_model(self) -> str:
        return self._model
    
    @property
    def model(self) -> str | None:
        return self._model
    
    def free(self) -> None:
        self._model = ""

    def run_inference(self, text: str, conditioning: TTSConditioning, stream: bool = False) -> list[bytes]:
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
        
        return list(self._elevenlabs_client.generate(
            text = text,
            voice=voice,
            model = self._model,
            stream = False
        ))