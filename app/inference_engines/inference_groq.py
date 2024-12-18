from typing import Literal

import groq

from .inference_engine_base import InferenceEngineBase
from .. import tool_data
from .. import security
from .. import llm_data

class InferenceEngine(InferenceEngineBase):
    def __init__(self):
        """
        This class provides the interface to run inference via the groq API.
        """
        self._key_manager = security.SecretsManager()

        self._model = None

    def select_model(self, model: Literal["llama-3.3-70b-versatile", "llama-3.2-90b-vision-preview"]) -> None:
        key = self._key_manager.get_secret("groq_api_key")

        if not key:
            raise ValueError("Groq API key not found")
        
        self._groq_client = groq.Groq(
            api_key=key
        )

        self._model = model

    def run_inference(self, conversation: llm_data.Conversation, tools: list[tool_data.LLMTool] | None):
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

        formated_response = llm_data.LLMResponse()
        formated_response.response_to_LLMResponse_format(response)

        return formated_response
    
    def get_current_model(self) -> str | None:
        return self._model