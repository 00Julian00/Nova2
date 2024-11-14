"""
Description: This script manages interactions with LLMs.
"""

#How to handle tool calls:
#Have an event system. You can subscribe to events.
#A tool call is not only returned to the caller, but also triggers the event system.

import groq

from .security import KeyManager

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
        Defines a tool that can be used by the LLM.
        """

        self.name = name
        self.description = description
        self.parameters = parameters

class LLMToolCallParameter:
    def __init__(
                self,
                name: str,
                value: str #The value is always a string. Casting needs to be handled by the tool that is executed. Alternativly leave the type ambigous and look up the type in the tool's parameter definition.
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

        response = self._groq_client.chat.completions.create(
            model=model,
            messages=conversation,
            tools=tools
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