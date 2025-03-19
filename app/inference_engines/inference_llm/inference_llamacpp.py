import multiprocessing
import atexit

from llama_cpp import Llama

from .inference_base_llm import InferenceEngineBaseLLM
from ...tool_data import *
from ...llm_data import *
from ...helpers import suppress_output

class InferenceEngineLlamaCPP(InferenceEngineBaseLLM):
    def __init__(self) -> None:
        """
        This class provides the interface to run inference via Llama cpp.
        """
        super().__init__()

        self.is_local = True

        self._model : Llama = None

        self._temp = 0
        self._max_tokens = 0

        atexit.register(self.free) # Safely clean up the model on exit

    def initialize_model(self, conditioning: LLMConditioning) -> None:
        self.free()

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

    def run_inference(self, conversation: Conversation, tools: List[LLMTool] | None) -> LLMResponse:
        conversation = conversation.to_list()

        # Check if tools were parsed
        if not tools or len(tools) == 0:
            response = self._model.create_chat_completion_openai_v1(
                messages=conversation,
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
                messages=conversation,
                tools=tool_list,
                temperature=self._temp,
                max_tokens=self._max_tokens
            )

        formated_response = LLMResponse()
        formated_response.from_dict(response)

        return formated_response
    
    def free(self) -> None:
        try:
            del self._model
        except:
            pass # Somewhere in Llamacpp an execption occurs in __del__. This try except just isolates that bug