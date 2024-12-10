"""
Description: This script manages interactions with LLMs.
"""

import groq
from transformers import AutoTokenizer

from .security import SecretsManager
from .tool_manager import LLMTool, LLMToolCall, ToolManager

class LLMResponse:
    def __init__(
                self,
                message: str | None = None,
                tool_calls: LLMToolCall | None = None,
                used_tokens: int = 0
                ) -> None:
        
        """
        Stores necessary LLM response information.
        """

        self.message = message
        self.tool_calls = tool_calls
        self.used_tokens = used_tokens

class LLMManager:
    def __init__(self) -> None:
        self._key_manager = SecretsManager()
        self._tool_manager = ToolManager()

        key = self._key_manager.get_secret("groq_api_key")

        if not key:
            raise ValueError("Groq API key not found")

        self._groq_client = groq.Groq(
            api_key=key
        )

    #TODO: Add support for vision models.
    #TODO: Add dynamic model selection to avoid rate limits.
    def prompt_llm(
                self,
                instruction: str | None,
                conversation: list[dict],
                tools: list[LLMTool] | None,
                model: str
                ) -> LLMResponse:

        """
        Prompts the LLM and returns the response.
        Instruction can be added as system prompt. Parse None to skip.
        """
        
        if instruction != "" and instruction is not None:
            conversation.append({"role": "system", "content": instruction})

        if tools:
            tools_json = self._tool_manager.convert_tool_list_to_json(tools)

        response = self._groq_client.chat.completions.create(
            model=model,
            messages=conversation,
            tools=tools_json
        )
        
        return self.construct_response(response)
        
    def construct_response(self, llm_response: dict) -> LLMResponse:
        """
        Constructs the LLMResponse object including tool calls from the LLM response.
        """

        if "error" in llm_response:
            raise RuntimeError(llm_response["error"]["message"])
        
        response = LLMResponse()
        if llm_response.choices[0].message.content:
            response.message = llm_response.choices[0].message.content

        if llm_response.choices[0].message.tool_calls:
            response.tool_calls = llm_response.choices[0].message.tool_calls

        response.used_tokens = llm_response.usage.total_tokens

        return response
    
    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        tokenizer = AutoTokenizer.from_pretrained(model)
        return len(tokenizer.tokenize(text))