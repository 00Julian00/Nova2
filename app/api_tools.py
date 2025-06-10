"""
Description: Implements the tool API.
"""

from pathlib import Path

from Nova2.app.context_manager import ContextManager
from Nova2.app.context_data import ContextDatapoint, ContextSource_ToolResponse
from Nova2.app.api_implementation import NovaAPI as APIImpl
from Nova2.app.interfaces import (
    ContextBase,
    STTConditioningBase,
    LLMConditioningBase,
    TTSConditioningBase,
    ContextGeneratorBase,
)

class NovaAPI(APIImpl):
    """
    Primary API to interact with the Nova system.
    """
    def add_to_context(self, name: str, content: str, tool_call_id: str) -> None: # type: ignore
        dp: ContextDatapoint = ContextDatapoint(
            source=ContextSource_ToolResponse(name=name, id=tool_call_id),
            content=content
        )
        ContextManager().add_to_context(datapoint=dp)

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