"""
Description: Holds all data required to run inference on LLMs.
"""

from typing import Literal
import json
from dataclasses import dataclass, field

from Nova2.app.tool_data import LLMToolCall, LLMToolCallParameter

from Nova2.app.interfaces import (
    LLMConditioningBase,
    MemoryConfigBase,
    MessageBase,
    LLMResponseBase,
    ConversationBase
)

class LLMConditioning(LLMConditioningBase):
    def __init__(
                self,
                model: str,
                inference_engine: str,
                filter_thinking_process: bool = True,
                temperature: float = 1.0,
                max_completion_tokens: int = 1024,
                add_default_sys_prompt: bool = True,
                **kwargs
                ) -> None:
        self.model = model
        self.inference_engine = inference_engine
        self.filter_thinking_process = filter_thinking_process
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.add_default_sys_prompt = add_default_sys_prompt
        self.kwargs = kwargs

@dataclass
class MemoryConfig(MemoryConfigBase):
    retrieve_memories: bool = True
    num_results: int = 2
    search_area: int = 2
    cosine_threshold: float = 0.6

class Message(MessageBase):
    def __init__(
            self,
            author: Literal["user", "assistant", "system", "tool"],
            content: str,
            **kwargs
            ) -> None:
        self._allowed_roles = ["user", "assistant", "system", "tool"]

        if (author not in self._allowed_roles):
            raise TypeError(f"Author must one of {self._allowed_roles}.")

        self.author = author
        self.content = content

        if author == "tool" :
            if "name" not in kwargs.keys():
                raise Exception("Keyword 'name' required for author 'tool'.")
            if "tool_call_id" not in kwargs.keys():
                raise Exception("Keyword 'tool_call_id' required for author 'tool'.")
            
            self.name = kwargs["name"]
            self.tool_call_id = kwargs["tool_call_id"]

@dataclass
class LLMResponse(LLMResponseBase):
    message: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    used_tokens: int = 0

    def from_dict(self, llm_response: dict) -> None:
        """
        Constructs the LLMResponse object including tool calls from the LLM response.

        Arguments:
            llm_response (dict): The response from the LLM that will be converted.
        """
        if "error" in llm_response:
            raise RuntimeError(llm_response["error"]["message"])
            
        if llm_response.choices[0].message.content: # type: ignore
            self.message = llm_response.choices[0].message.content # type: ignore

        if llm_response.choices[0].message.tool_calls: # type: ignore
            for tool_call in llm_response.choices[0].message.tool_calls: # type: ignore
                params = []

                args = json.loads(tool_call.function.arguments)

                keys = list(args.keys())
                values = list(args.values())

                for i, _ in enumerate(keys):
                    params.append(
                        LLMToolCallParameter(
                            name=keys[i],
                            value=values[i]
                        )
                    )

                self.tool_calls.append(
                    LLMToolCall(
                        name=tool_call.function.name,
                        parameters=params,
                        id=tool_call.id
                    )
                )

        self.used_tokens = llm_response.usage.total_tokens # type: ignore

    def to_message(self) -> Message:
        return Message(author="assistant", content=self.message)

class Conversation(ConversationBase):
    def __init__(
            self,
            conversation: list[Message] = []
            ) -> None:
        self._conversation = conversation
        self._allowed_roles = ["user", "assistant", "system"]

    def add_message(self, message: MessageBase) -> None:
        self._conversation.append(message) # type: ignore

    def add_messages(self, messages: list[MessageBase]) -> None:
        self._conversation += messages

    def delete_newest(self, author: Literal["user", "assistant", "system", None] = None) -> None:
        if author == None:
            del self._conversation[-1]
        else:
            #Itterate through the list from the back and find the first one with a matching author
            for i, message in enumerate(reversed(self._conversation)):
                if message.author == author: # type: ignore
                    del self._conversation[i]
                    break

    def delete_all_from(self, author: Literal["user", "assistant", "system"]) -> None:
        for i, message in enumerate(reversed(self._conversation)):
            if message.author == author: # type: ignore
                del self._conversation[i]

    def get_newest(self, author: Literal["user", "assistant", "system", None] = None) -> Message | None:
        """
        Get the newest message. If an author is parsed, the newest message of that author will be returned.

        Arguments:
            author (Literal["user", "assistant", "system", None]): An optional parameter. The author whose newest message will be returned.

        Returns:
            Message | None: The newest message (from the author). None if there are no messages in the conversation or no messages from the specified author.
        """
        if len(self._conversation) == 0:
            return None
        
        if not author:
            msg = self._conversation[-1]
            return self._conversation[-1] # type: ignore
        
        for i, message in enumerate(reversed(self._conversation)):
            if message.author == author: # type: ignore
                return message # type: ignore
 
    def to_list(self) -> list[dict]:
        conversation = []

        for message in self._conversation:
            if message.author == "tool": # type: ignore
                conversation.append({"role": "tool", "name": message.name, "content": message.content, "tool_call_id": message.tool_call_id}) # type: ignore
            else:
                conversation.append({"role": message.author, "content": message.content}) # type: ignore

        return conversation
    
    def from_list(self,
            conversation: list[dict]
            ) -> None:
        self._conversation = []

        for message in conversation:
            self._conversation.append(Message(author=message["role"], content=message["content"]))