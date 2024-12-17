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

        self._inference_engine.select_model(model=model)

        response = self._inference_engine.run_inference(conversation=conversation, tools=tools)

        return response
    
    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        tokenizer = AutoTokenizer.from_pretrained(model)
        return len(tokenizer.tokenize(text))