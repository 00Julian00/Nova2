"""
Description: Acts as the base API for the main API and the tool API.
"""

from pathlib import Path
import logging

from app.tts_manager import *
from app.llm_manager import *
from app.audio_manager import *
from app.transcriptor import *
from app.context_manager import *
from app.context_manager import *
from app.inference_engines import *
from app.security_manager import *

from app.inference_engines.inference_tts.inference_zonos import InferenceEngineZonos

class NovaAPI:
    def __init__(self) -> None:
        """
        Novas API.
        """
        self._tts = TTSManager()
        self._llm = LLMManager()
        self._stt = VoiceAnalysis()

        self._context = ContextManager()
        self._context_data = ContextManager()
        self._player = AudioPlayer()
        self._tools = ToolManager()
        self._security = SecretsManager()

        self._max_ctx_length = 25
        self._append_sys_prompt = True

        logging.getLogger().setLevel(logging.CRITICAL)

    def configure_transcriptor(self, conditioning: TranscriptorConditioning) -> None:
        """
        Configure the transcriptor.
        """
        self._stt.configure(conditioning=conditioning)

    def configure_llm(self, inference_engine: InferenceEngineBaseLLM, conditioning: LLMConditioning) -> None:
        """
        Configure the LLM system.
        """
        self._llm.configure(inference_engine=inference_engine, conditioning=conditioning)

    def configure_tts(self, inference_engine: InferenceEngineBaseTTS, conditioning: TTSConditioning) -> None:
        """
        Configure the Text-to-Speech system.
        """
        self._tts.configure(inference_engine=inference_engine, conditioning=conditioning)

    def apply_config_all(self) -> None:
        """
        Updates the configuration of the transcriptor, LLM and TTS systems. Also loads the chosen models into memory.
        """
        self._tts.apply_config()
        self._llm.apply_config()
        self._stt.apply_config()

    def apply_config_llm(self) -> None:
        """
        Updates the configuration of the LLM system. Also loads the chosen models into memory.
        """
        self._llm.apply_config()

    def apply_config_tts(self) -> None:
        """
        Updates the configuration of the TTS system. Also loads the chosen models into memory.
        """
        self._tts.apply_config()

    def apply_config_transcriptor(self) -> None:
        """
        Updates the configuration of the trabscriptor system. Also loads the chosen models into memory.
        """
        self._stt.apply_config()

    def load_tools(self, load_internal_tools: bool = True, **kwargs) -> List[LLMTool]:
        """
        Load all tools in the tool folder into memory and make them ready for calling.

        Arguments:
            load_internal_tools (bool): Wether the set of internal tools should be loaded. If set to false, the LLM loses access to some core functionality like renaming voices in the database or creating new memories.
            include (List[string]): Which tools should be loaded. Incompatible with "exclude".
            exclude (List[string]): Which tools should not be loaded. Incompatible with "include".

        Returns:
            List[LLMTool]: A list of loaded tools that can be parsed when running the LLM to give it access to these tools.
        """
        return self._tools.load_tools(load_internal=load_internal_tools, **kwargs)
    
    def execute_tool_calls(self, llm_response: LLMResponse) -> None:
        """
        Execute the tools that were called by the LLM.

        Arguments:
            tool_calls (List[LLMToolCall]): The tool calls from the LLM that should be executed.
        """
        self._tools.execute_tool_call(tool_calls=llm_response.tool_calls)

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

    def start_transcriptor(self) -> ContextGenerator:
        """
        Start the transcriptor. The transcriptor will start to listen to the microphone audio.

        Returns:
            ContextSource: The context source representing the transcriptor. Needs to be binded to record its data in "bind_context_source()".
        """
        return ContextGenerator(self._stt.start())

    def bind_context_source(self, source: ContextGenerator) -> None:
        """
        Bind a context source. The data of a context source will only be recorded after beeing bound.
        """
        self._context.record_data(source)

    def get_context(self) -> Context:
        """
        Get the current context.
        """
        return self._context_data.get_context_data()
    
    def set_ctx_limit(self, ctx_limit: int) -> None:
        """
        Limit how many datapoints will be stored in context. This does not include memory.
        Setting it to 0 will impose no limit, but the context will surpass the LLMs context window at some point.
        Limit is 25 by default.
        """
        self._context_data.ctx_limit = ctx_limit

    def add_to_context(self, name: str, content: str, id: str) -> None:
        """
        Add a response from the tool to the context.

        Arguments:
            name (str): The name of the tool. Should match the name given in metadata.json.
            content (str): The message that should be added to the context
        """
        dp = ContextDatapoint(
            source=ContextSource_ToolResponse(
                name=name,
                id=id
            ),
            content=content
        )

        ContextManager().add_to_context(datapoint=dp)
    
    def add_llm_response_to_context(self, response: LLMResponse) -> None:
        """
        Add LLMResponse to the context.
        """
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

    def is_playing_audio(self) -> bool:
        """
        Checks wether the audio player is currently playing any audio.
        """
        return self._player.is_playing()
    
    def clone_voice(self, mp3file: Path, name: str) -> None:
        """
        Clones a voice from an mp3 file and stores it in /data/voices.
        After cloning, it can be used with the Zonos inference engine.

        Arguments: 
            mp3file (Path): The mp3 file containing a few seconds of speech of the voice that will be cloned.
            name (str): What the voice should be called.
        """
        zonos = InferenceEngineZonos()
        zonos.clone_voice(audio_dir=str(mp3file), name=name)

    def huggingface_login(self, overwrite: bool = False, token: str = ""):
        """
        Attempt to log into huggingface which is required to access restricted repos.
        Raises an exception if the login fails.
        
        Arguments:
            overwrite (bool): If true, "token" will overwrite the value stored in the database. If false, the database will remain unchanged and "token" will be used to attempt a login, if provided.
            token (str): If provided, this token will be used to log in.
        """
        self._security.huggingface_login(overwrite=overwrite, token=token)

    def edit_secret(self, name: Secrets, value: str) -> None:
        """
        Edit a secret, like an API key in the database. The value will be encrypted before it is stored.

        Arguments:
            name (Secrets): Which of the secrets to edit.
            value (str): The new value of the secret.
        """
        self._security.edit_secret(name=name, key=value)