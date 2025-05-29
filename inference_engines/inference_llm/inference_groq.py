import os

import groq

from Nova2.app.interfaces import LLMInferenceEngineBase
from Nova2.app.tool_data import *
from Nova2.app.security_manager import *
from Nova2.app.llm_data import *

class InferenceEngineGroq(LLMInferenceEngineBase):
    def __init__(self):
        """
        This class provides the interface to run inference via the groq API.
        """
        self._key_manager = SecretsManager()

        self._model: str = ""

        super().__init__()

        self.is_local = False

    def initialize_model(self, conditioning: LLMConditioning) -> None: # type: ignore
        key = os.getenv("GROQ_API_KEY")

        if not key:
            raise ValueError("Groq API key not found")
        
        self._groq_client = groq.Groq(
            api_key=key
        )

        self._model = conditioning.model

    def run_inference(self, conversation: Conversation, tools: list[LLMTool] | None) -> LLMResponse: # type: ignore
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
    
    def free(self) -> None:
        del self._groq_client
    
    @property
    def model(self) -> str:
        return self._model