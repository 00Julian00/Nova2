"""
Description: This script manages interactions with LLMs.
"""

from transformers import AutoTokenizer

from Nova2.app.tool_data import LLMTool
from Nova2.app.database_manager import MemoryEmbeddingDatabaseManager
from Nova2.app.interfaces import LLMInferenceEngineBase
from Nova2.app.llm_data import LLMConditioning, LLMResponse, Conversation, MemoryConfig, Message
from Nova2.app.context_data import Context
from Nova2.app.library_manager import LibraryManager
from Nova2.app.inference_engine_manager import InferenceEngineManager

class LLMManager:
    def __init__(self) -> None:
        """
        This class provides the interface for LLM interaction.
        """
        self._conditioning: LLMConditioning = None # type: ignore
        self._conditioning_dirty = None

        self._library = LibraryManager()
        self._inference_engine_manager = InferenceEngineManager()

    def configure(self, conditioning: LLMConditioning) -> None:
        """
        Configure the LLM system.
        """
        self._conditioning_dirty = conditioning

    def apply_config(self) -> None:
        """
        Applies the configuration and loads the model into memory.
        """
        if self._conditioning_dirty is None:
            raise Exception("Failed to initialize LLM. No LLM conditioning provided.")

        self._conditioning = self._conditioning_dirty

        self._inference_engine: LLMInferenceEngineBase = self._inference_engine_manager.request_engine(
            self._conditioning.inference_engine,
            "LLM"
            ) # type: ignore

        self._inference_engine.initialize_model(self._conditioning)

    def prompt_llm(
                self,
                conversation: Conversation | Context,
                tools: list[LLMTool] | None = None,
                memory_config: MemoryConfig | None = None,
                instruction: str | None = None
                ) -> LLMResponse:
        """
        Run inference on an LLM.

        Arguments:
            instruction (str | None): Instruction is added as a system prompt.
            conversation (Conversation | Context): The conversation that the LLM will base its response on. Can be tyoe Conversation or type Context.
            tools (list[LLMTool] | None): The tools the LLM has access to.
            model (str): The model that should be used for inference.
            perform_rag (bool): Wether to search for addidtional data in the memory database based on the newest user message.

        Returns:
            LLMResponse: The response of the LLM. Also includes tool calls.
        """
        if type(conversation) == Context:
            conv: Conversation = conversation.to_conversation()
        else:
            conv: Conversation = conversation # type: ignore

        if self._conditioning.add_default_sys_prompt:
            prompt = self._library.retrieve_datapoint("prompt_library", "default_sys_prompt")
            conv.add_message(Message(author="system", content=prompt)) # type: ignore

        if instruction != "" and instruction is not None:
            conv.add_message(Message(author="system", content=instruction))

        # Can not process an empty conversation. Add dummy data
        if len(conv._conversation) == 0:
            conv.add_message(Message(author="system", content="You are a helpful assistant."))

        if memory_config and memory_config.retrieve_memories:
            db = MemoryEmbeddingDatabaseManager()

            text = conv.get_newest("user").content # type: ignore

            text_split = text.split(". ")

            results = ""

            for sentence in text_split:
                retrieved = db.search_semantic(
                                            text=sentence,
                                            num_of_results=memory_config.num_results,
                                            search_area=memory_config.search_area,
                                            cosine_threshold=memory_config.cosine_threshold
                                            )
                
                if retrieved:
                    for block in retrieved:
                        for sent in block:
                            results += sent

                    results += "|"

            if results != "": # Don't add anything if there are no search results
                conv.add_message(
                    Message(author="system", content=f"Information that is potentially relevant to the conversation: {results}. This information was retrieved from the database.")
                    )

        response = self._inference_engine.run_inference(conversation=conv, tools=tools) # type: ignore

        # Split at "</think>"
        if self._conditioning.filter_thinking_process:
            resp_clean = ""

            split = response.message.split("</think>")
            if len(split) > 1:
                resp_clean = split[1]
            else:
                resp_clean = response.message

            response.message = resp_clean.strip()

        return response # type: ignore

    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        tokenizer = AutoTokenizer.from_pretrained(model)
        return len(tokenizer.tokenize(text))