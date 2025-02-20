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
from app.tool_manager import *

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

        self._tools = ToolManager()

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

    def apply_config(self) -> None:
        """
        Updates the configuration of the transcriptor, LLM and TTS systems. Also loads the chosen models into memory.
        """
        self._tts.apply_config()
        self._llm.apply_config()
        self._stt.apply_config()

    def load_tools(self) -> List[LLMTool]:
        """
        Load all tools in the tool folder into memory and make them ready for calling.

        Returns:
            List[LLMTool]: A list of loaded tools that can be parsed when running the LLM to give it access to these tools.
        """
        return self._tools.load_tools()
    
    def execute_tool_calls(self, tool_calls: List[LLMToolCall]) -> None:
        """
        Execute the tools that were called by the LLM.

        Arguments:
            tool_calls (List[LLMToolCall]): The tool calls from the LLM that should be executed.
        """
        self._tools.execute_tool_call(tool_calls=tool_calls)

    def run_llm(self, conversation: Conversation, memory_config: MemoryConfig = None, tools: List[LLMTool] = None, instruction: str = "") -> LLMResponse:
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
        """
        Start the transcriptor. The transcriptor will start to listen to the microphone audio.

        Returns:
            ContextSource: The context source representing the transcriptor. Needs to be binded to record its data in "bind_context_source()".
        """
        return ContextSource(self._stt.start())

    def bind_context_source(self, source: ContextSource) -> None:
        """
        Bind a context source. The data of a context source will only be recorded after beeing bound.
        """
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