"""
Description: Holds all data required to run inference on LLMs.
"""

from typing import Literal
import json

from .tool_data import LLMToolCall, LLMToolCallParameter

class LLMConditioning:
    def __init__(
                self,
                model: str,
                temperature: float = 1.0,
                max_completion_tokens: int = 1024,
                add_default_sys_prompt: bool = True,
                **kwargs
                ) -> None:
        """
        Stores all values required for LLM conditioning.

        Arguments:
            model (str): The model name. Must a valid huggingface repo ID.
            file (str): The file to use from that repo. Must be GGUF format.
            temperature (float): The temperature to use for inference.
            max_completion_tokens (int): How many tokens the model is allowed to generate.
            add_default_sys_prompt (bool): Should an extra system prompt be added to the LLM that adds context about the Nova system?
        """
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.add_default_sys_prompt = add_default_sys_prompt
        self.kwargs = kwargs

class MemoryConfig:
    def __init__(
                self,
                retrieve_memories: bool = True,
                num_results: int = 2,
                search_area: int = 2,
                cosine_threshold: float = 0.6
                ) -> None:
        """
        Stores the settings that determines how memories are retrieved.

        Arguments:
            retrieve_memories (bool): Wether to search for memories in the database.
            num_results (int): The maximum amount of results that should be fed to the model.
            search_area (int): How much context around the search result should additionally be fed to the model.
            cosine_threshold (float): The similarity threshold a result must surpass to be utilized.
        """
        self.retrieve_memories = retrieve_memories
        self.num_results = num_results
        self.search_area = search_area
        self.cosine_threshold = cosine_threshold

class Message:
    def __init__(
            self,
            author: Literal["user", "assistant", "system", "tool"],
            content: str,
            name: str = "",
            tool_call_id: str = ""
            ) -> None:
        """
        Stores one message in a conversation.

        Parameter "name" and "tool_call_id" are only required if "author" == "tool"
        """
        self._allowed_roles = ["user", "assistant", "system", "tool"]

        if (author not in self._allowed_roles):
            raise TypeError(f"Author must one one of {self._allowed_roles}.")

        self.author = author
        self.content = content
        self.name = name
        self.tool_call_id = tool_call_id

class LLMResponse:
    def __init__(
                self,
                message: str = "",
                tool_calls: list[LLMToolCall] = [],
                used_tokens: int = 0
                ) -> None:
        """
        Stores necessary LLM response information.
        """
        self.message = message
        self.tool_calls = tool_calls
        self.used_tokens = used_tokens

    def from_dict(self, llm_response: dict) -> None:
        """
        Constructs the LLMResponse object including tool calls from the LLM response.

        Arguments:
            llm_response (dict): The response from the LLM that will be converted.
        """
        if "error" in llm_response:
            raise RuntimeError(llm_response["error"]["message"])
            
        if llm_response.choices[0].message.content:
            self.message = llm_response.choices[0].message.content

        if llm_response.choices[0].message.tool_calls:
            for tool_call in llm_response.choices[0].message.tool_calls:
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

        self.used_tokens = llm_response.usage.total_tokens

    def to_message(self) -> Message:
        """
        Formats the LLM reponse to a Message object.
        """
        return Message(author="assistant", content=self.message)

class Conversation:
    def __init__(self, conversation: list[Message] = []) -> None:
        """
        Stores the conversation between the LLM and the user.
        """
        self._conversation = conversation
        self._allowed_roles = ["user", "assistant", "system"]

    def add_message(self, message: Message) -> None:
        """
        Add one message to the conversation.

        Arguments:
            message (Message): The message that will be added.
        """
        self._conversation.append(message)

    def add_messages(self, messages: list[Message]) -> None:
        """
        Add multiple messages to the conversation.

        Arguments:
            messages (list[Message]): The messages that will be added.
        """
        self._conversation += messages

    def delete_newest(self, author: Literal["user", "assistant", "system", None] = None) -> None:
        """
        Delete the newest message in the conversation. If an author is parsed, the newest message with that author is deleted.

        Arguments:
            author (Literal["user", "assistant", "system", None]): An optional parameter. The author whose newest message will be deleted.
        """
        if author == None:
            del self._conversation[-1]
        else:
            #Itterate through the list from the back and find the first one with a matching author
            for i, message in enumerate(reversed(self._conversation)):
                if message.author == author:
                    del self._conversation[i]
                    break

    def delete_all_from(self, author: Literal["user", "assistant", "system"]) -> None:
        """
        Delete all messages from an author. Can be used to purge system prompts if behaviour should be overwritten.

        Arguments:
            author (Literal["user", "assistant", "system"]): The author from whom all messages should be deleted.
        """
        for i, message in enumerate(reversed(self._conversation)):
            if message.author == author:
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
            return self._conversation[-1]
        
        for i, message in enumerate(reversed(self._conversation)):
            if message.author == author:
                return message

    def to_list(self) -> list[dict]:
        """
        Convert the stored conversation to a format that can be parsed to the LLM.

        Returns:
            list[dict]: The formatted conversation.
        """
        conversation = []

        for message in self._conversation:
            if message.author == "tool":
                conversation.append({"role": "tool", "name": message.name, "content": message.content, "tool_call_id": message.tool_call_id})
            else:
                conversation.append({"role": message.author, "content": message.content})

        return conversation
    
    def from_list(self, conversation: list[dict]) -> None:
        """
        Convert the LLM format conversation into a conversation object. Overwrites the stored conversation.

        Arguments:
            converation (list[dict]): The conversation that should be converted and stored.
        """
        self._conversation = []

        for message in conversation:
            self._conversation.append(Message(author=message["role"], content=message["content"]))