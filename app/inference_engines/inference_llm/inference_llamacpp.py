import multiprocessing
import atexit

from llama_cpp import Llama

from Nova2.app.inference_engines.inference_llm.inference_base_llm import InferenceEngineBaseLLM
from Nova2.app.tool_data import *
from Nova2.app.llm_data import *
from Nova2.app.helpers import suppress_output, deprecated

class InferenceEngineLlamaCPP(InferenceEngineBaseLLM):
    def __init__(self) -> None:
        """
        This class provides the interface to run inference via Llama cpp.
        """
        super().__init__()

        self.is_local = True

        self._model: Llama | None = None
        self._conditioning: LLMConditioning | None = None

        self._temp = 0
        self._max_tokens = 0

        atexit.register(self.free) # Safely clean up the model on exit

    def initialize_model(self, conditioning: LLMConditioning) -> None:
        self.free()

        self._conditioning = conditioning

        if "ctx_size" in conditioning.kwargs:
            ctx_size = conditioning.kwargs["ctx_size"]
        else:
            ctx_size = 1024

        with suppress_output():
            self._model = Llama.from_pretrained(
                repo_id=conditioning.model,
                n_gpu_layers=-1,
                n_threads=multiprocessing.cpu_count(),
                flash_attn=True,
                filename=conditioning.kwargs["file"],
                n_ctx=ctx_size,
                verbose=False
            )

        self._temp = conditioning.temperature
        self._max_tokens = conditioning.max_completion_tokens

    def run_inference(self, conversation: Conversation, tools: list[LLMTool] | None) -> LLMResponse:
        assert self._model is not None
        
        conv = conversation.to_list()

        # Check if tools were parsed
        if not tools or len(tools) == 0:
            response = self._model.create_chat_completion_openai_v1(
                messages=conv,
                temperature=self._temp,
                max_tokens=self._max_tokens
            )
        else:
            # Prepare the tools
            tool_list = []

            for tool in tools:
                tool_list.append(tool.to_dict())

            # Run inference
            response = self._model.create_chat_completion_openai_v1(
                messages=conv,
                tools=tool_list,
                temperature=self._temp,
                max_tokens=self._max_tokens
            )

        formated_response = LLMResponse()
        formated_response.from_dict(response) # type: ignore

        return formated_response
    
    def free(self) -> None:
        try:
            del self._model
        except:
            pass # Somewhere in Llamacpp an execption occurs in __del__. This try except just isolates that bug

    @deprecated(warning=f"Function 'get_current_model' is deprecated and will be removed in a future update. Use the property 'model' instead.")
    def get_current_model(self) -> str:
        if not self._conditioning:
            return ""
        return self._conditioning.model
    
    @property
    def model(self) -> str | None:
        if not self._conditioning:
            return ""
        return self._conditioning.model