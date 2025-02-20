"""
Description: This script manages interactions with LLMs.
"""

from transformers import AutoTokenizer

from .tool_data import LLMTool
from .tool_manager import *
from .database_manager import MemoryEmbeddingDatabaseManager
from .inference_engines.inference_base_llm import InferenceEngineBase
from .llm_data import *

class LLMManager:
    def __init__(self) -> None:
        """
        This class provides the interface for LLM interaction.
        """
        self._inference_engine = None
        self._conditioning = None

        self._inference_engine_dirty = None
        self._conditioning_dirty = None
        
    def configure(self, inference_engine: InferenceEngineBase, conditioning: LLMConditioning) -> None:
        """
        Configure the LLM system.
        """
        if inference_engine._type != "LLM":
            raise TypeError("Inference engine must be of type \"LLM\"")
        
        self._inference_engine_dirty = inference_engine
        self._conditioning_dirty = conditioning

    def apply_config(self) -> None:
        """
        Applies the configuration and loads the model into memory.
        """
        if self._inference_engine_dirty is None:
            raise Exception("Failed to initialize LLM. No inference engine provided.")
        
        if self._conditioning_dirty is None:
            raise Exception("Failed to initialize LLM. No LLM conditioning provided.")

        self._inference_engine = self._inference_engine_dirty
        self._conditioning = self._conditioning_dirty

        self._inference_engine.initialize_model(self._conditioning)

    def prompt_llm(
                self,
                conversation: Conversation,
                tools: list[LLMTool] | None,
                memory_config: MemoryConfig,
                instruction: str | None = None
                ) -> LLMResponse:
        """
        Run inference on an LLM.

        Arguments:
            instruction (str | None): Instruction is added as a system prompt.
            conversation (Conversation): The conversation that the LLM will base its response on.
            tools (list[LLMTool] | None): The tools the LLM has access to.
            model (str): The model that should be used for inference.
            perform_rag (bool): Wether to search for addidtional data in the memory database based on the newest user message.

        Returns:
            LLMResponse: The response of the LLM. Also includes tool calls.
        """
        if instruction != "" and instruction is not None:
            conversation.add_message(Message(author="system", content=instruction))

        if memory_config.retrieve_memories:
            db = MemoryEmbeddingDatabaseManager()
            db.open()

            text = conversation.get_newest("user").content

            text_split = text.split(". ")

            results = ""

            for sentence in text_split:
                retrieved = db.search_semantic(
                                            text=sentence,
                                            num_of_results=memory_config.num_results,
                                            search_area=memory_config.search_area,
                                            cosine_threshold=memory_config.cosine_threshold
                                            )
                
                for block in retrieved:
                    for sent in block:
                        results += sent

                results += "|"
            
            db.close()

            if results != "": # Don't add anything if there are no search results
                conversation.add_message(
                    Message(author="system", content=f"Information that is potentially relevant to the conversation: {results}. This information was retrieved from the database.")
                    )

        return self._inference_engine.run_inference(conversation=conversation, tools=tools)
    
    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        tokenizer = AutoTokenizer.from_pretrained(model)
        return len(tokenizer.tokenize(text))