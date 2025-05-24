"""
Description: Provides a base class for all LLM inference engines to ensure a consistent structure and documentation.
"""

from Nova2.app.llm_data import LLMConditioning, Conversation, LLMResponse
from Nova2.app.tool_data import LLMTool

class InferenceEngineBaseLLM:
    """
    Provides a base class for all LLM inference engines to ensure a consistent structure.
    """
    def __init__(self):
        self._type = "LLM"
        self.is_local = False

    def initialize_model(self, conditioning: LLMConditioning):
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again.
        """
        pass

    def initialize_model_async(self):
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again. Runs async.
        """
        pass

    def is_model_ready(self):
        """
        Checks if the engine is ready to run inference.
        """
        pass
    
    def get_current_model(self) -> str:
        """
        Which model is currently loaded.
        """
        return "" # Dirty fix to make pylance happy. Will be completly revamped in 2.0 anyways

    def free(self):
        """
        Frees the VRAM/RAM. The model can not be used anymore after it was freed. It needs to be loaded again by calling select_model().
        """
        pass

    def run_inference(self, conversation: Conversation, tools: list[LLMTool] | None) -> LLMResponse:
        """
        Prompt the LLM and get an answer from it.
        """
        return LLMResponse()