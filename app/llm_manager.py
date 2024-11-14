"""
Description: This script manages interactions with LLMs.
"""

import groq

from .security import KeyManager

class LLMResponse:
    def __init__(
                self,
                message: str,
                tool_calls: list[dict] | None = None,
                used_tokens: int = 0
                ) -> None:

        """
        This class is used to store only the necessary information from the LLM response in a structured format.

        Contents:
            message (str): The message from the LLM.
            tool_calls (list[dict] | None): The tool calls from the LLM.
            used_tokens (int): The number of tokens used by the LLM (includes both input and output).
        """

        self.message = message
        self.tool_calls = tool_calls
        self.used_tokens = used_tokens

class LLMToolParameter:
    def __init__(
                self,
                name: str,
                description: str,
                type: str,
                required: bool
                ) -> None:

        """
        Defines a parameter for a tool.
        """

        self.name = name
        self.description = description
        self.type = type
        self.required = required

class LLMTool:
    def __init__(
                self,
                name: str,
                description: str,
                parameters: list[LLMToolParameter]
                ) -> None:

        """
        Defines a tool that can be used in the LLM.
        """

        self.name = name
        self.description = description
        self.parameters = parameters

class LLMToolCallParameter:
    def __init__(
                self,
                name: str,
                value: str
                ) -> None:

        """
        Defines a parameter for a tool call.
        """

        self.name = name
        self.value = value

class LLMToolCall:
    def __init__(
                self,
                name: str,
                parameters: list[LLMToolCallParameter]
                ) -> None:

        """
        Defines a tool call made by the LLM.
        """

        self.name = name
        self.parameters = parameters

class LLMManager:
    def __init__(self) -> None:
        self._key_manager = KeyManager()

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
                instruction: str,
                conversation: list[dict],
                tools: list[LLMTool] | None,
                model: str
                ) -> LLMResponse:

        """
        Prompts the LLM and returns the response.
        Instruction can be added as system prompt. Parse empty string to skip.
        """
        
        if instruction != "":
            conversation.append({"role": "system", "content": instruction})

        if tools:
            response = self._groq_client.chat.completions.create(
                model=model,
                messages=conversation,
                tools=tools
            )
        else:
            response = self._groq_client.chat.completions.create(
                model=model,
                messages=conversation
            )

        if "error" in response:
            raise RuntimeError(response["error"]["message"])
        
        #Construct the appropriate LLMResponse object.
        if response.choices[0].message != "None":
            return LLMResponse(
                message=response.choices[0].message.content,
                tool_calls=response.choices[0].message.tool_calls,
                used_tokens=response.usage.total_tokens
            )
        else:
            return LLMResponse(
                message=response.choices[0].message.content,
                used_tokens=response.usage.total_tokens
            )