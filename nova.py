"""
Description: Acts as the API for the entire Nova system.
"""

from app.tts_manager import *
from app.llm_manager import *
from app.audio_manager import *
from app.transcriptor import *
from app.context_manager import *
from app.memory_manager import *

from app.inference_engines import *

class Nova:
    def __init__(self) -> None:
        """
        API to easily construct an AI assistant using the Nova framework.
        """
        self._tts = TTSManager()
        self._llm = LLMManager()
        self._stt = VoiceAnalysis()

        self._context = ContextManager()
        self._player = AudioPlayer()

        self._mem_manager = MemoryManager()

    def configure_transcriptor(self, conditioning: TranscriptorConditioning) -> None:
        """
        Configure the transcriptor.
        """
        self._stt.configure(conditioning=conditioning)

    def configure_llm(self, inference_engine: InferenceEngineBase, conditioning: LLMConditioning) -> None:
        """
        Configure the LLM system.
        """
        self._llm.configure(inference_engine=inference_engine, conditioning=conditioning)

    def configure_tts(self, inference_engine: InferenceEngineBase, conditioning: TTSConditioning) -> None:
        """
        Configure the Text-to-Speech system.
        """
        self._tts.configure(inference_engine=inference_engine, conditioning=conditioning)

    def mem_check(self) -> MemCheckResult:
        if self._tts._inference_engine_dirty.is_local:
            tts_model_mem = self._mem_manager.estimate_memory_requirement(self._tts._conditioning_dirty.model)
        else:
            tts_model_mem = 0

        if self._llm._inference_engine_dirty.is_local:
            llm_model_mem = self._mem_manager.estimate_memory_requirement(self._llm._conditioning_dirty.model)
        else:
            llm_model_mem = 0
        
        stt_model_mem = self._mem_manager.estimate_memory_requirement(self._stt._conditioning_dirty.model)

        vram_required = tts_model_mem + llm_model_mem
        ram_required = 0

        if self._stt._conditioning_dirty.device == "cpu":
            ram_required = stt_model_mem
        else:
            vram_required += stt_model_mem

        return self._mem_manager.construct_mem_check_result(ram_required=ram_required, vram_required=vram_required)

    def apply_config(self) -> None:
        """
        Updates the configuration of the transcriptor, LLM and TTS systems. Also loads the chosen models into memory.
        """
        self._tts.apply_config()
        self._llm.apply_config()
        self._stt.apply_config()

    def run_llm(self, conversation: Conversation, memory_config: MemoryConfig = None, tools: list[LLMTool] = None, instruction: str = "") -> LLMResponse:
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
        return self._llm.prompt_llm(conversation=conversation, tools=tools, memory_config=memory_config, instruction=instruction)

    def run_tts(self, text: str) -> AudioData:
        """
        Run inference on the TTS.

        Arguments:
            text (str): The text that should be turned into speech.

        Returns:
            AudioData: The resulting audio data that can be played by the audio player.
        """
        return self._tts.run_inference(text=text)

    def start_transcriptor(self) -> ContextSource:
        return ContextSource(self._stt.start())

    def bind_context_source(self, source: ContextSource) -> None:
        self._context.record_data(source)

    def play_audio(self, audio_data: AudioData) -> None:
        """
        Use the built in audio player to play audio. Only accepts an AudioData object.
        """
        self._player.play_audio(audio_data)

    def wait_for_audio_playback_end(self) -> None:
        """
        Halts the code execution until the audio player is done playing the current audio.
        """
        while self._player.is_playing():
            time.sleep(0.1)