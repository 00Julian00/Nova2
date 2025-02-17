"""
Description: Acts as the API for the entire Nova system.
"""
import app.tts_manager as tts
import app.llm_manager as llm
import app.audio_manager as player
import app.transcriptor as transcriptor
import app.context_manager as context

class MemCheckResult:
    pass

class Nova:
    def __init__(self) -> None:
        """
        API to easily construct an AI assistant using the Nova framework.
        """
        self._tts = tts.TTSManager()
        self._llm = llm.LLMManager()
        self._stt = transcriptor.VoiceAnalysis()

        self._context = context.ContextManager()
        self._player = player.AudioPlayer()

    def configure_transcriptor(self, conditioning: transcriptor.TranscriptorConditioning) -> None:
        """
        Configure the transcriptor.
        """
        self._stt.configure(conditioning=conditioning)

    def configure_llm(self, inference_engine: llm.InferenceEngineBase, conditioning: llm.LLMConditioning) -> None:
        """
        Configure the LLM system.
        """
        self._llm.configure(inference_engine=inference_engine, conditioning=conditioning)

    def configure_tts(self, inference_engine: tts.InferenceEngineBase, conditioning: tts.TTSConditioning) -> None:
        """
        Configure the Text-to-Speech system.
        """
        self._tts.configure(inference_engine=inference_engine, conditioning=conditioning)

    def mem_check(self) -> MemCheckResult:
        raise NotImplementedError()

    def apply_config(self) -> None:
        """
        Updates the configuration of the transcriptor, LLM and TTS systems. Also loads the chosen models into memory.
        """
        self._tts.apply_config()
        self._llm.apply_config()
        self._stt.apply_config()

    def run_llm(self, conversation: llm.Conversation, memory_config: llm.MemoryConfig = None, tools: list[llm.LLMTool] = None, instruction: str = "") -> llm.LLMResponse:
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

    def run_tts(self, text: str) -> tts.AudioData:
        """
        Run inference on the TTS.

        Arguments:
            text (str): The text that should be turned into speech.

        Returns:
            AudioData: The resulting audio data that can be played by the audio player.
        """
        return self._tts.run_inference(text=text)

    def start_transcriptor(self) -> transcriptor.Listener:
        return transcriptor.Listener(self._stt.start())

    def record_context(self, listener: transcriptor.Listener) -> None:
        self._context.record_data(listener)

    def play_audio(self, audio_data: player.AudioData) -> None:
        self._player.play_audio(audio_data)