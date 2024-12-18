"""
Description: This script manages interactions with LLMs.
"""

from transformers import AutoTokenizer

from .tool_data import LLMTool
from .database_manager import MemoryEmbeddingDatabaseManager
from .inference_engines import inference_groq
from .llm_data import LLMResponse, Message, Conversation

class LLMManager:
    def __init__(self) -> None:
        """
        This class provides the interface for LLM interaction.
        """
        self._inference_engine = inference_groq.InferenceEngine()

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

        if perform_rag:
            db = MemoryEmbeddingDatabaseManager()
            db.open()
            retieved = db.search_semantic( #TODO: Split
                text=conversation.get_newest("user").content,
                num_of_results=2,
                search_area=2
                )
            
            db.close()

            if retieved: # Don't add anything if there are no search results
                conversation.add_message(
                    Message(author="system", content=f"Information that is potentially relevant to the conversation: {retieved}. This information was retrieved from the database.")
                    )

        if self._inference_engine.get_current_model() != model:
            self._inference_engine.select_model(model=model)

        response = self._inference_engine.run_inference(conversation=conversation, tools=tools)

        return response
    
    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        tokenizer = AutoTokenizer.from_pretrained(model)
        return len(tokenizer.tokenize(text))