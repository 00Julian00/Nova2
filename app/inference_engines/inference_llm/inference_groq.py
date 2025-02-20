import groq

from .inference_base_llm import InferenceEngineBaseLLM
from ...tool_data import *
from ...security import *
from ...llm_data import *

class InferenceEngineGroq(InferenceEngineBaseLLM):
    def __init__(self):
        """
        This class provides the interface to run inference via the groq API.
        """
        self._key_manager = SecretsManager()

        self._model = None

        super().__init__()

        self.is_local = False

    def initialize_model(self, conditioning: LLMConditioning) -> None:
        key = self._key_manager.get_secret("groq_api_key")

        if not key:
            raise ValueError("Groq API key not found")
        
        self._groq_client = groq.Groq(
            api_key=key
        )

        self._model = conditioning.model

    def run_inference(self, conversation: Conversation, tools: list[LLMTool] | None):
        conversation = conversation.conversation_to_llm_format()
        
        # Check if tools were parsed
        if not tools or len(tools) == 0:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=conversation
            )
        else:
            # Prepare the tools
            tool_list = []

            for tool in tools:
                tool_list.append(tool.to_json())

            # Run inference
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=conversation,
                tools=tool_list
            )

        formated_response = LLMResponse()
        formated_response.response_to_LLMResponse_format(response)

        return formated_response
    
    def get_current_model(self) -> str | None:
        return self._model