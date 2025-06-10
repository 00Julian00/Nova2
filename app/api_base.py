"""
Description: Abstract definition of the tool API. Used for DI.
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List

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
)

class APIAbstract(ABC):
    @abstractmethod
    def configure_transcriptor(self, conditioning: STTConditioningBase) -> None:
        """
        Configure the transcriptor.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_llm(self,conditioning: LLMConditioningBase) -> None:
        """
        Configure the LLM system.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_tts(self, conditioning: TTSConditioningBase) -> None:
        """
        Configure the Text-to-Speech system.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_config_all(self) -> None:
        """
        Updates the configuration of the transcriptor, LLM and TTS systems. Also loads the chosen models into memory.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_config_llm(self) -> None:
        """
        Updates the configuration of the LLM system. Also loads the chosen models into memory.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_config_tts(self) -> None:
        """
        Updates the configuration of the TTS system. Also loads the chosen models into memory.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_config_transcriptor(self) -> None:
        """
        Updates the configuration of the trabscriptor system. Also loads the chosen models into memory.
        """
        raise NotImplementedError

    @abstractmethod
    def run_llm(self, conversation: ConversationBase, memory_config: MemoryConfigBase = None, tools: List[LLMToolBase] = None, instruction: str = "") -> LLMResponseBase: # type: ignore
        """
        Run inference on the LLM.

        Arguments:
            conversation (Conversation): A conversation to use. Can be retrieved from context.
            memory_config (MemoryConfig): How should memories be retieved? If none is provided, no memories will be retrieved.
            tools (list[LLMTool]): A list of tools the LLM can access.
            instruction (str): An additional instruction to give to the LLM.

        Returns:
            LLMResponse: A response object containing all relevant information about what the LLM has responded.
        """
        raise NotImplementedError

    @abstractmethod
    def run_tts(self, text: str) -> AudioDataBase:
        """
        Run inference on the TTS.

        Arguments:
            text (str): The text that should be turned into speech.

        Returns:
            AudioData: The resulting audio data that can be played by the audio player.
        """
        raise NotImplementedError

    @abstractmethod
    def start_transcriptor(self) -> ContextGeneratorBase:
        """
        Start the transcriptor. The transcriptor will start to listen to the microphone audio.

        Returns:
            ContextGenerator: An object yielding context data from the transcriptor.
        """
        raise NotImplementedError

    @abstractmethod
    def bind_context_source(self, source: ContextGeneratorBase) -> None:
        """
        Bind a context source. The data of a context source will only be recorded after beeing bound.
        """
        raise NotImplementedError

    @abstractmethod
    def get_context(self) -> ContextBase:
        """
        Get the current context.
        """
        raise NotImplementedError

    @abstractmethod
    def set_context(self, context: ContextBase) -> None:
        """
        Overwrites the stored context data.
        """
        raise NotImplementedError

    @abstractmethod
    def set_ctx_limit(self, ctx_limit: int) -> None:
        """
        Limit how many datapoints will be stored in context. This does not include memory.
        Setting it to 0 will impose no limit, but the context will surraise NotImplementedError the LLMs context window at some point.
        Limit is 25 by default.
        """
        raise NotImplementedError

    @abstractmethod
    def add_to_context(self, name: str, content: str, id: str) -> None:
        """
        Add a response from a tool to the context.

        Arguments:
            name (str): The name of the tool. Should match the name given in metadata.json.
            content (str): The message that should be added to the context.
            id (str): The unique identifier for the specific tool call that generated this context.
        """
        raise NotImplementedError

    @abstractmethod
    def add_datapoint_to_context(self, datapoint: ContextDatapointBase) -> None:
        """
        Adds a pre-constructed ContextDatapoint to the context.
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_active_context_file(self, file_name: str = "") -> None:
        """
        Changes the current context data to the one stored in the specified file.
        Saves the currently active context data to the context file before changing.
        
        Arguments:
            file_name (str): The name of the file to load the context data from (without the .ctx extension). Defaults to a random UUID.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_active_context_file(self) -> str:
        """
        Returns the currently active context file.

        Returns:
            str: The path to the currently active context file.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_all_context_files(self) -> list[str]:
        """
        Returns all context files in the context folder.

        Returns:
            list[str]: A list of all context files in the context folder.
        """
        raise NotImplementedError
    
    @abstractmethod
    def rename_context_file(self, old_name: str, new_name: str) -> None:
        """
        Renames the currently active context file.

        Arguments:
            old_name (str): The current name of the context file (without the .ctx extension).
            new_name (str): The new name for the context file (without the .ctx extension).
        """
        raise NotImplementedError

    @abstractmethod
    def is_context_initialized(self) -> bool:
        """
        Checks if a context file is set and initialized.
        It is only possible to modify the context if it is initialized.

        Returns:
            bool: True if the context is initialized, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def play_audio(self, audio_data: AudioDataBase) -> None:
        """
        Use the built in audio player to play audio. Only accepts an AudioData object.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_audio_playback_end(self) -> None:
        """
        Halts the code execution until the audio player is done playing the current audio.
        """
        raise NotImplementedError

    @abstractmethod
    def is_playing_audio(self) -> bool:
        """
        Checks wether the audio player is currently playing any audio.
        """
        raise NotImplementedError

    @abstractmethod
    def clone_voice(self, engine: str, mp3file: Path, name: str) -> None:
        """
        Clones a voice from an mp3 file and stores it (location determined by implementation).
        After cloning, it should be usable by a compatible TTS inference engine.

        Arguments:
            engine (str): The name of the TTS inference engine that should be used to clone the voice.
            mp3file (Path): The mp3 file containing a few seconds of speech of the voice that will be cloned.
            name (str): What the voice should be called.
        """
        raise NotImplementedError

    @abstractmethod
    def huggingface_login(self):
        """
        Attempt to log into huggingface which is required to access restricted repos.
        Should raise an exception if the login fails. Uses the credentials stored in the .env file.
        """
        raise NotImplementedError
    
    @abstractmethod
    def stay_alive(self, condition: bool = True) -> None:
        """
        Keeps the application running until the condition is False.
        This is useful, because none of the internal logic of Nova2 inherently
        keeps the program alive.

        Arguments:
            condition (bool): The application will stay alive, as long as this is True. It is recommended to parse a lambda expression. Defaults to True.
        """
        raise NotImplementedError