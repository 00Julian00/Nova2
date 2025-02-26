"""
Description: This script collects all data provided from the voice analysis and stores them in long and short term context memory.
"""

from threading import Thread
import time

import torch

from .context_data_manager import ContextDataManager
from .context_data import ContextSource, ContextSourceList
from .database_manager import VoiceDatabaseManager
from .transcriptor_data import Word

class ContextManager:
    source_list = ContextSourceList()
    def __init__(self) -> None:
        """
        Prepares context data provided by a listener and stores them in the context file.
        """
        self._context_data_manager = ContextDataManager()
        self._voice_database_manager = VoiceDatabaseManager()

        self._context_recorder = Thread(target=self._record_context, daemon=True)
        self._context_recorder.start()

    def record_data(self, source: ContextSource) -> None:
        """
        Begins to listen to the source and record the data.
        """
        ContextManager.source_list.add(context_source=source)

    def _record_context(self) -> None:
        """
        Stores the context of all bound context sources.
        """
        while True:
            datapoint = ContextManager.source_list.get_next()
            if not datapoint:
                continue

            self._context_data_manager.add_to_context(datapoint=datapoint)

            time.sleep(0.1)

    def _word_array_to_string(self, word_array: list[Word]) -> str:
        text = ""
        for word in word_array:
            text += word.text
        return text
    
    def _take_average_embedding(self, embeddings: list[torch.FloatTensor]) -> torch.FloatTensor:
        return torch.mean(torch.stack(embeddings), dim=0)