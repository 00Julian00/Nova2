"""
Description: This script manages interactions with LLMs.
"""

from typing import Literal

import groq
from transformers import AutoTokenizer

from .security import SecretsManager
from .tool_manager import LLMTool, LLMToolCall, ToolManager
from .database_manager import MemoryEmbeddingDatabaseManager

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
                conversation: Conversation,
                tools: list[LLMTool] | None,
                model: str,
                perform_rag: bool = False
                ) -> LLMResponse:

        """
        Prompts the LLM and returns the response.
        Instruction can be added as system prompt. Parse None to skip.
        """

        if instruction != "" and instruction is not None:
            conversation.add_message(Message(author="system", content=instruction))

        if perform_rag:
            db = MemoryEmbeddingDatabaseManager()
            db.open()
            retieved = db.search_semantic( #TODO: Split
                text=conversation.get_newest("user").content,
                num_of_results=2,
                search_area=2
                )
            
            db.close()

            if retieved: #Don't add anything if there are no search results
                conversation.add_message(
                    Message(author="system", content=f"Information that is potentially relevant to the conversation: {retieved}. This information was retrieved from the database.")
                    )

        conversation = conversation.conversation_to_llm_format()

        if tools:
            tools_json = self._tool_manager.convert_tool_list_to_json(tools)

            response = self._groq_client.chat.completions.create(
                model=model,
                messages=conversation,
                tools=tools_json
            )

        else:
            response = self._groq_client.chat.completions.create(
                model=model,
                messages=conversation
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