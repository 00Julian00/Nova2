"""
Description: Provides a base class for all LLM inference engines to ensure a consistent structure and documentation.
"""

class InferenceEngineBaseLLM:
    """
    Provides a base class for all LLM inference engines to ensure a consistent structure.
    """
    def __init__(self):
        self._type = "LLM"
        self.is_local = False

    def initialize_model():
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again.
        """
        pass

    def initialize_model_async():
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again. Runs async.
        """
        pass

    def is_model_ready():
        """
        Checks if the engine is ready to run inference.
        """
        pass

    def get_current_model():
        """
        Which model is currently loaded.
        """
        pass

    def free():
        """
        Frees the VRAM/RAM. The model can not be used anymore after it was freed. It needs to be loaded again by calling select_model().
        """
        pass

    def run_inference():
        """
        Prompt the LLM and get an answer from it.
        """
        pass