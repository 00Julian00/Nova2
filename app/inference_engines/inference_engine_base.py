"""
Description: Provides a base class for all inference engines to ensure a consistent structure.
"""

from .. import llm_data

class InferenceEngineBase:
    """
    Provides a base class for all inference engines to ensure a consistent structure.
    """

    def select_model() -> None:
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again.
        """
        pass

    def select_model_async() -> None:
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again. Runs async.
        """
        pass

    def is_model_ready() -> bool:
        """
        Checks if the engine is ready to run inference.
        """
        pass

    def free() -> None:
        """
        Frees the VRAM/RAM. The model can not be used anymore after it was freed. It needs to be loaded again by calling select_model().
        """
        pass

    def run_inference() -> llm_data.LLMResponse:
        """
        Prompt the LLM and get an answer from it.
        """
        pass