"""
Description: Implements the main API.
"""

from pathlib import Path
import logging
import time

from .tts_manager import TTSManager
from .llm_manager import LLMManager
from .audio_manager import AudioPlayer
from .transcriptor import VoiceAnalysis
from .context_manager import ContextManager, ContextDatapoint
from .inference_engines.inference_tts.inference_zonos import InferenceEngineZonos
from .security_manager import SecretsManager
from .api_base import APIAbstract
from .context_data import ContextSource_Assistant, ContextGenerator
from .tool_manager import ToolManager
from .security_data import Secrets
from .shared_types import (
    TranscriptorConditioningBase,
    InferenceEngineLLMBase,
    InferenceEngineTTSBase,
    LLMConditioningBase,
    TTSConditioningBase,
    LLMToolBase,
    LLMResponseBase,
    ContextBase,
    ContextDatapointBase,
    ConversationBase,
    MemoryConfigBase,
    AudioDataBase,
    ContextGeneratorBase,
    ContextSourceBase,
)

class NovaAPI(APIAbstract):
    """
    Primary API to interact with the Nova system.
    """
    def __init__(self) -> None:
        self._tts = TTSManager()
        self._llm = LLMManager()
        self._stt = VoiceAnalysis()

        self._context = ContextManager()
        self._context_data = ContextManager()
        self._player = AudioPlayer()
        self._tools = ToolManager()
        self._security = SecretsManager()

        logging.getLogger().setLevel(logging.CRITICAL)

    def configure_transcriptor(self, conditioning: TranscriptorConditioningBase) -> None:
        self._stt.configure(conditioning=conditioning)

    def configure_llm(self, inference_engine: InferenceEngineLLMBase, conditioning: LLMConditioningBase) -> None:
        self._llm.configure(inference_engine=inference_engine, conditioning=conditioning)

    def configure_tts(self, inference_engine: InferenceEngineTTSBase, conditioning: TTSConditioningBase) -> None:
        self._tts.configure(inference_engine=inference_engine, conditioning=conditioning)

    def apply_config_all(self) -> None:
        self._tts.apply_config()
        self._llm.apply_config()
        self._stt.apply_config()

    def apply_config_llm(self) -> None:
        self._llm.apply_config()

    def apply_config_tts(self) -> None:
        self._tts.apply_config()

    def apply_config_transcriptor(self) -> None:
        self._stt.apply_config()

    def load_tools(self, load_internal_tools: bool = True, **kwargs) -> list[LLMToolBase]:
        return self._tools.load_tools(load_internal=load_internal_tools, **kwargs)
    
    def execute_tool_calls(self, llm_response: LLMResponseBase) -> None:
        self._tools.execute_tool_call(tool_calls=llm_response.tool_calls)

    def run_llm(self, conversation: ConversationBase, memory_config: MemoryConfigBase = None, tools: list[LLMToolBase] = None, instruction: str = "") -> LLMResponseBase:
        return self._llm.prompt_llm(conversation=conversation, tools=tools, memory_config=memory_config, instruction=instruction)

    def run_tts(self, text: str) -> AudioDataBase:
        return self._tts.run_inference(text=text)

    def start_transcriptor(self) -> ContextGeneratorBase:
        return ContextGenerator(self._stt.start())

    def bind_context_source(self, source: ContextGeneratorBase) -> None:
        self._context.record_data(source)

    def get_context(self) -> ContextBase:
        return self._context_data.get_context_data()
    
    def set_context(self, context: ContextBase) -> None:
        self._context_data._overwrite_context(context.data_points)
    
    def set_ctx_limit(self, ctx_limit: int) -> None:
        self._context_data.ctx_limit = ctx_limit

    def add_to_context(self, source: ContextSourceBase, content: str) -> None:
        dp = ContextDatapoint(
            source=source,
            content=content
        )
        ContextManager().add_to_context(datapoint=dp)

    def add_datapoint_to_context(self, datapoint: ContextDatapointBase) -> None:
        ContextManager().add_to_context(datapoint=datapoint)
    
    def add_llm_response_to_context(self, response: LLMResponseBase) -> None:
        if len(response.tool_calls) > 0:
            for tool_call in response.tool_calls:
                self._context_data.add_to_context(
                    ContextDatapoint(
                        source=ContextSource_Assistant(),
                        content=f"Called tool \"{tool_call.name}\""
                    ))
        else:
            self._context_data.add_to_context(
                ContextDatapoint(
                    source=ContextSource_Assistant(),
                    content=response.message
                ))

    def play_audio(self, audio_data: AudioDataBase) -> None:
        self._player.play_audio(audio_data)

    def wait_for_audio_playback_end(self) -> None:
        while self._player.is_playing():
            time.sleep(0.1)

    def is_playing_audio(self) -> bool:
        return self._player.is_playing()
    
    def clone_voice(self, mp3file: Path, name: str) -> None:
        zonos = InferenceEngineZonos()
        zonos.clone_voice(audio_dir=str(mp3file), name=name)

    def huggingface_login(self, overwrite: bool = False, token: str = ""):
        self._security.huggingface_login(overwrite=overwrite, token=token)

    def edit_secret(self, name: Secrets, value: str) -> None:
        self._security.edit_secret(name=name, key=value)