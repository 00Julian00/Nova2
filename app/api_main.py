"""
Description: Implements the main API.
"""

from pathlib import Path
import logging
import time

from Nova2.app.tts_manager import TTSManager
from Nova2.app.llm_manager import LLMManager
from Nova2.app.audio_manager import AudioPlayer
from Nova2.app.stt_manager import VoiceAnalysis
from Nova2.app.context_manager import ContextManager, ContextDatapoint
from Nova2.app.inference_engine_manager import InferenceEngineManager
from Nova2.app.security_manager import SecretsManager
from Nova2.app.api_base import APIAbstract
from Nova2.app.context_data import ContextSource_Assistant, ContextGenerator
from Nova2.app.tool_manager import ToolManager
from Nova2.app.interfaces import (
    STTConditioningBase,
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
        self._engine_manager = InferenceEngineManager()

        logging.getLogger().setLevel(logging.CRITICAL)

    def configure_transcriptor(self, conditioning: STTConditioningBase) -> None:
        self._stt.configure(conditioning=conditioning) # type: ignore

    def configure_llm(self, conditioning: LLMConditioningBase) -> None:
        self._llm.configure(conditioning=conditioning) # type: ignore

    def configure_tts(self, conditioning: TTSConditioningBase) -> None:
        self._tts.configure(conditioning=conditioning) # type: ignore

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
        return self._tools.load_tools(load_internal=load_internal_tools, **kwargs) # type: ignore
    
    def execute_tool_calls(self, llm_response: LLMResponseBase) -> None:
        self._tools.execute_tool_call(tool_calls=llm_response.tool_calls) # type: ignore

    def run_llm(self, conversation: ConversationBase, memory_config: MemoryConfigBase = None, tools: list[LLMToolBase] = None, instruction: str = "") -> LLMResponseBase: # type: ignore
        return self._llm.prompt_llm(conversation=conversation, tools=tools, memory_config=memory_config, instruction=instruction) # type: ignore

    def run_tts(self, text: str) -> AudioDataBase:
        return self._tts.run_inference(text=text)

    def start_transcriptor(self) -> ContextGeneratorBase:
        return ContextGenerator(self._stt.start())

    def bind_context_source(self, source: ContextGeneratorBase) -> None:
        self._context.record_data(source) # type: ignore

    def is_context_initialized(self) -> bool:
        return self._context.is_context_initialized()

    def set_active_context_file(self, file_name: str = "") -> None:
        self._context.set_active_context_file(file_name=file_name)

    def get_all_context_files(self) -> list[str]:
        return self._context.get_all_context_files()
    
    def rename_context_file(self, old_name: str, new_name: str) -> None:
        self._context.rename_context_file(old_name=old_name, new_name=new_name)

    def get_active_context_file(self) -> str:
        return self._context.get_active_context_file()

    def get_context(self) -> ContextBase:
        return self._context_data.get_context_data()
    
    def set_context(self, context: ContextBase) -> None:
        self._context_data._overwrite_context(context.data_points) # type: ignore
     
    def set_ctx_limit(self, ctx_limit: int) -> None:
        self._context_data.ctx_limit = ctx_limit

    def add_to_context(self, source: ContextSourceBase, content: str) -> None: # type: ignore
        dp = ContextDatapoint(
            source=source, # type: ignore
            content=content
        )
        ContextManager().add_to_context(datapoint=dp)

    def add_datapoint_to_context(self, datapoint: ContextDatapointBase) -> None:
        ContextManager().add_to_context(datapoint=datapoint) # type: ignore
    
    def add_llm_response_to_context(self, response: LLMResponseBase) -> None:
        if len(response.tool_calls) > 0: # type: ignore
            for tool_call in response.tool_calls: # type: ignore
                self._context_data.add_to_context(
                    ContextDatapoint(
                        source=ContextSource_Assistant(),
                        content=f"Called tool \"{tool_call.name}\""
                    ))
        else:
            self._context_data.add_to_context(
                ContextDatapoint(
                    source=ContextSource_Assistant(),
                    content=response.message # type: ignore
                ))

    def play_audio(self, audio_data: AudioDataBase) -> None:
        self._player.play_audio(audio_data) # type: ignore

    def wait_for_audio_playback_end(self) -> None:
        while self._player.is_playing():
            time.sleep(0.1)

    def is_playing_audio(self) -> bool:
        return self._player.is_playing()
    
    def clone_voice(self, engine: str, mp3file: Path, name: str) -> None:
        eng = self._engine_manager.request_engine(name=engine, eng_type="TTS")
        eng.clone_voice(audio_dir=str(mp3file), name=name) # type: ignore

    def huggingface_login(self):
        self._security.huggingface_login()

    def stay_alive(self, condition: bool = True) -> None:
        while condition:
            time.sleep(0.1) # Don't block the GIL