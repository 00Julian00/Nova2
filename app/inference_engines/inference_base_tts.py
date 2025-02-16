"""
Description: Provides a base class for all TTS inference engines to ensure a consistent structure and documentation.
"""

class InferenceEngineBase:
    """
    Provides a base class for all TTS inference engines to ensure a consistent structure.
    """

    def initialize_model():
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again.
        """
        pass

    def select_model_async():
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again. Runs async.
        """
        pass

    def select_voice():
        """
        Set the voice to be used.
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
        Get the spoken text from the TTS.
        """
        pass