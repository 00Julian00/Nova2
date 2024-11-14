"""
Description: This script controls the execution flow of all the parts of the Nova system.
"""

from time import sleep

from .transcriptor import VoiceAnalysis
from .context_manager import ContextManager

class AssistantPipeline:
    def __init__(self) -> None:
        pass

    def start(self) -> None:
        self._transcriptor = VoiceAnalysis(microphone_index=3, speculative=False, whisper_model="deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", voice_boost=10.0, language="de", verbose=True)

        self._context_manager = ContextManager(voice_analysis=self._transcriptor.start())
        self._context_manager.start()

        while True:
            sleep(1) #Keep the main thread alive.

    def close(self) -> None:
        self._transcriptor.close()
        self._context_manager.close()

    def __del__(self) -> None:
        self.close()
