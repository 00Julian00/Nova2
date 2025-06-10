"""
Description: Implements the tool API.
"""

from pathlib import Path
import logging
import time

from Nova2.app.tts_manager import TTSManager
from Nova2.app.llm_manager import LLMManager
from Nova2.app.audio_manager import AudioPlayer
from Nova2.app.stt_manager import VoiceAnalysis
from Nova2.app.context_manager import ContextManager
from Nova2.app.context_data import ContextDatapoint, ContextSource_ToolResponse
from Nova2.app.api_base import APIAbstract
from Nova2.app.interfaces import (
    LLMToolBase,
    LLMResponseBase,
    ContextBase,
    ContextDatapointBase,
    ConversationBase,
    MemoryConfigBase,
    AudioDataBase,
    STTConditioningBase,
    LLMConditioningBase,
    TTSConditioningBase,
    ContextGeneratorBase,
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

        logging.getLogger().setLevel(logging.CRITICAL)

    def run_llm(self, conversation: ConversationBase, memory_config: MemoryConfigBase = None, tools: list[LLMToolBase] = None, instruction: str = "") -> LLMResponseBase: # type: ignore
        return self._llm.prompt_llm(conversation=conversation, tools=tools, memory_config=memory_config, instruction=instruction) # type: ignore

    def run_tts(self, text: str) -> AudioDataBase:
        return self._tts.run_inference(text=text)
    
    def is_context_initialized(self) -> bool:
        return self._context.is_context_initialized()
    
    def get_active_context_file(self) -> str:
        return self._context.get_active_context_file()

    def get_context(self) -> ContextBase:
        return self._context_data.get_context_data()
    
    def add_to_context(self, name: str, content: str, tool_call_id: str) -> None: # type: ignore
        dp: ContextDatapoint = ContextDatapoint(
            source=ContextSource_ToolResponse(name=name, id=tool_call_id),
            content=content
        )
        ContextManager().add_to_context(datapoint=dp)

    def add_datapoint_to_context(self, datapoint: ContextDatapointBase):
        ContextManager().add_to_context(datapoint=datapoint) # type: ignore

    def play_audio(self, audio_data: AudioDataBase) -> None:
        self._player.play_audio(audio_data) # type: ignore

    def wait_for_audio_playback_end(self) -> None:
        while self._player.is_playing():
            time.sleep(0.1)

    def is_playing_audio(self) -> bool:
        return self._player.is_playing()
    

    def configure_transcriptor(self, conditioning: STTConditioningBase) -> None:
        raise NotImplementedError()
    
    def configure_llm(self, conditioning: LLMConditioningBase) -> None:
        raise NotImplementedError()

    def configure_tts(self, conditioning: TTSConditioningBase) -> None:
        raise NotImplementedError()
    
    def apply_config_all(self) -> None:
        raise NotImplementedError()
    
    def apply_config_llm(self) -> None:
        raise NotImplementedError()
    
    def apply_config_tts(self) -> None:
        raise NotImplementedError()
    
    def apply_config_transcriptor(self) -> None:
        raise NotImplementedError()
    
    def bind_context_source(self, source: ContextGeneratorBase) -> None:
        raise NotImplementedError()
    
    def set_active_context_file(self, file_name: str = "") -> None:
        raise NotImplementedError()

    def get_all_context_files(self) -> list[str]:
        raise NotImplementedError()
    
    def rename_context_file(self, old_name: str, new_name: str) -> None:
        raise NotImplementedError()
    
    def clone_voice(self, engine: str, mp3file: Path, name: str) -> None:
        raise NotImplementedError()
    
    def huggingface_login(self):
        raise NotImplementedError()
    
    def set_context(self, context: ContextBase) -> None:
        raise NotImplementedError()
    
    def set_ctx_limit(self, ctx_limit: int) -> None:
        raise NotImplementedError()
    
    def start_transcriptor(self) -> ContextGeneratorBase:
        raise NotImplementedError()
    
    def stay_alive(self, condition: bool = True) -> None:
        raise NotImplementedError()