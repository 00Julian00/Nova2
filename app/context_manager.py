"""
Description: This script collects all data provided from the voice analysis and stores them in long and short term context memory.
"""

from threading import Thread
import time
from pathlib import Path
import json

import torch

from .context_data import *
from .database_manager import VoiceDatabaseManager
from .transcriptor_data import Word

class ContextManager:
    source_list = ContextGeneratorList()
    context_data = []
    def __init__(self) -> None:
        """
        Prepares context data provided by a listener and stores them in the context file.
        """
        self._voice_database_manager = VoiceDatabaseManager()

        self._context_file = Path(__file__).parent.parent / "data" / "context.json"
        ContextManager.context_data = self._prepare_context_data()

        self.ctx_limit = 25

        self._context_recorder = Thread(target=self._record_context, daemon=True)
        self._context_recorder.start()

    def record_data(self, source: ContextGenerator) -> None:
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

            self.add_to_context(datapoint=datapoint)

            time.sleep(0.1)

    def add_to_context(self, datapoint: ContextDatapoint) -> None:
        """
        Adds content to the context.json file.

        Arguments:
            datapoint (ContextDatapoint): The datapoint that will be added to the context.
        """
        ContextManager.context_data.append(datapoint.format())

        if self.ctx_limit > 0:
            ContextManager.context_data = ContextManager.context_data[-self.ctx_limit:]

        self.save_context_data()

    def save_context_data(self) -> None:
        """
        Saves the context data to the context.json file.
        """
        with open(self._context_file, 'w') as file:
            json.dump(ContextManager.context_data, file, indent=4)

    def get_context_data(self) -> Context:
        """
        Reconstructs a Context object from the stored context.

        Returns:
            Context: The context data stored in memory.
        """
        sources = ContextSourceBase.get_all_sources()

        datapoints = []

        for datapoint in ContextManager.context_data:
            source_instance = None
            # Find the correct source
            for source in sources:
                if source.__name__ == datapoint["source"]["type"] and source.__name__ != "ContextSourceBase":
                    if "metadata" in datapoint["source"]:
                        metadata = datapoint["source"]["metadata"]
                        source_instance = source(**metadata)
                    else:
                        source_instance = source()
                    break
            
            if not source_instance:
                dp = datapoint["source"]["type"]
                raise Exception(f"Got unknown context source {dp}")

            datapoint_instance = ContextDatapoint(
                source=source_instance,
                content=datapoint["content"]
                )
            
            datapoint_instance.timestamp = datapoint["timestamp"]
            datapoints.append(datapoint_instance)

        return Context(datapoints)

    def _prepare_context_data(self) -> List[dict]:
        """
        Creates a context.json file if it does not exist. Returns the contents of the context.json file.
        
        Returns:
            list[dict]: The contents of the context.json file.
        """
        if not self._context_file.exists():
            with open(self._context_file, 'w') as file:
                json.dump([], file)

        with open(self._context_file, 'r') as file:
            context_data = json.load(file)

        return context_data

    def _word_array_to_string(self, word_array: list[Word]) -> str:
        text = ""
        for word in word_array:
            text += word.text
        return text
    
    def _take_average_embedding(self, embeddings: list[torch.FloatTensor]) -> torch.FloatTensor:
        return torch.mean(torch.stack(embeddings), dim=0)