import groq

from Nova2.app.inference_engines.inference_llm.inference_base_llm import InferenceEngineBaseLLM
from Nova2.app.tool_data import *
from Nova2.app.security_manager import *
from Nova2.app.security_data import Secrets
from Nova2.app.llm_data import *
from Nova2.app.helpers import deprecated

class InferenceEngineGroq(InferenceEngineBaseLLM):
    def __init__(self):
        """
        This class provides the interface to run inference via the groq API.
        """
        self._key_manager = SecretsManager()

        self._model: str = ""

        super().__init__()

        self.is_local = False

    def initialize_model(self, conditioning: LLMConditioning) -> None:
        key = self._key_manager.get_secret(Secrets.GROQ_API)

        if not key:
            raise ValueError("Groq API key not found")
        
        self._groq_client = groq.Groq(
            api_key=key
        )

        self._model = conditioning.model

    def run_inference(self, conversation: Conversation, tools: list[LLMTool] | None) -> LLMResponse:
        conv = conversation.to_list()

        # Check if tools were parsed
        if not tools or len(tools) == 0:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=conv # type: ignore
            )
        else:
            # Prepare the tools
            tool_list = []

            for tool in tools:
                tool_list.append(tool.to_dict())

            # Run inference
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=conv, # type: ignore
                tools=tool_list
            )

        formated_response = LLMResponse()
        formated_response.from_dict(response) # type: ignore

        return formated_response
    
    @deprecated(warning=f"Function 'get_current_model' is deprecated and will be removed in a future update. Use the property 'model' instead.")
    def get_current_model(self) -> str:
        return self._model
    
    @property
    def model(self) -> str | None:
        return self._model