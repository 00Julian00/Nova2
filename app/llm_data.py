"""
Description: Holds all data required to run inference on LLMs.
"""

from typing import Literal

from .tool_data import LLMToolCall

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

    def response_to_LLMResponse_format(self, llm_response: dict) -> None:
        """
        Constructs the LLMResponse object including tool calls from the LLM response.
        """

        if "error" in llm_response:
            raise RuntimeError(llm_response["error"]["message"])
            
        if llm_response.choices[0].message.content:
            self.message = llm_response.choices[0].message.content

        if llm_response.choices[0].message.tool_calls:
            self.tool_calls = llm_response.choices[0].message.tool_calls

        self.used_tokens = llm_response.usage.total_tokens

class Message:
    def __init__(
            self,
            author: Literal["user", "assistant", "system"],
            content: str
            ) -> None:
        
        self._allowed_roles = ["user", "assistant", "system"]

        if (author not in self._allowed_roles):
            raise TypeError(f"Author must one one of {self._allowed_roles}.")

        self.author = author
        self.content = content

class Conversation:
    def __init__(self, conversation: list[Message] = []) -> None:
        self._conversation = conversation
        self._allowed_roles = ["user", "assistant", "system"]

    def add_message(self, message: Message) -> None:
        self._conversation.append(message)

    def add_messages(self, messages: list[Message]) -> None:
        self._conversation += messages

    def delete_last(self, author: Literal["user", "assistant", "system", None] = None):
        """
        Delete the last message in the conversation. If an author is parsed, the last message with that author is deleted.
        """

        if author == None:
            del self._conversation[-1]
        else:
            #Itterate through the list from the back and find the first one with a matching author
            for i, message in enumerate(reversed(self._conversation)):
                if message.author == author:
                    del self._conversation[i]
                    break

    def delete_all_from(self, author: Literal["user", "assistant", "system"]):
        """
        Delete all messages from an author. Can be used to purge system prompts if behaviour should be overwritten.
        """

        for i, message in enumerate(reversed(self._conversation)):
            if message.author == author:
                del self._conversation[i]

    def get_newest(self, author: Literal["user", "assistant", "system", None] = None) -> Message | None:
        """
        Returns the newest message. If an author is parsed, the newest message of that author will be returned.
        Returns none if the conversation is empty, or there are no messages from the specified author.
        """

        if len(self._conversation) == 0:
            return None
        
        if not author:
            return self._conversation[-1]
        
        for i, message in enumerate(reversed(self._conversation)):
            if message.author == author:
                return message

    def conversation_to_llm_format(self) -> list[dict]:
        """
        Convert the stored conversation to a format that can be parsed to the LLM
        """
        conversation = []

        for message in self._conversation:
            conversation.append({"role": message.author, "content": message.content})

        return conversation
    
    def llm_format_to_conversation(self, conversation: list[dict]) -> None:
        """
        Convert the LLM format conversation into a conversation object. Overwrites the stored conversation.
        """
        self._conversation = []

        for message in conversation:
            self._conversation.append(Message(author=message["role"], content=message["content"]))