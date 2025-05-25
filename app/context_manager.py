"""
Description: This script collects all data provided from the voice analysis and stores them in long and short term context memory.
"""

from threading import Thread
import time
from pathlib import Path
import json
import atexit

import torch

from Nova2.app.context_data import ContextDatapoint, ContextSource, ContextSource_Voice, Context, ContextGenerator, ContextGeneratorList
from Nova2.app.stt_data import Word
from Nova2.app.helpers import Singleton

class ContextManager(Singleton):
    def __init__(self, saving_interval: int = 120) -> None:
        """
        Prepares context data provided by a listener and stores them in the context file.

        Arguments:
            saving_interval (int): The interval in seconds at which the context data will be saved to a file. The context will also be saved when the program is closed.
        """
        self.source_list = ContextGeneratorList()
        self.context_data = []
        self._previous_context_data = []
        self.saving_interval = saving_interval

        self._context_file = Path(__file__).parent.parent / "data" / "context.json"
        self.context_data = self._prepare_context_data()

        self.ctx_limit = 25

        # Save the context data to the disk when the program is terminated
        atexit.register(self.save_context_data)

        self._saving_thread = Thread(target=self._periodic_save, daemon=True)
        self._saving_thread.start()

        self._context_recorder = Thread(target=self._record_context, daemon=True)
        self._context_recorder.start()

    def record_data(self, source: ContextGenerator) -> None:
        """
        Begins to listen to the source and record the data.
        """
        self.source_list.add(context_source=source)

    def _record_context(self) -> None:
        """
        Stores the context of all bound context sources.
        """
        while True:
            datapoint = self.source_list.get_next()
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
        self.context_data.append(datapoint.to_dict())

        if self.ctx_limit > 0:
            self.context_data = self.context_data[-self.ctx_limit:]

    def _overwrite_context(self, context: list[ContextDatapoint]) -> None:
        """
        Overwrites the entire context. Use with caution.

        Arguments:
            context (List[ContextDatapoint]): The data the context will be overwritten with.
        """
        self.context_data = []

        for datapoint in context:
            self.add_to_context(datapoint=datapoint)

    def _periodic_save(self):
        """
        Periodically saves the context data to the context.json file.
        """
        while True:
            time.sleep(self.saving_interval)
            self.save_context_data()

    def save_context_data(self) -> None:
        """
        Saves the context data to the context.json file.
        """
        if self.context_data != self._previous_context_data:
            with open(self._context_file, 'w') as file:
                json.dump(self.context_data, file, indent=4)

        self._previous_context_data = self.context_data

    def get_context_data(self) -> Context:
        """
        Reconstructs a Context object from the stored context.

        Returns:
            Context: The context data stored in memory.
        """
        sources = ContextSource.get_all_sources()

        datapoints = []

        for datapoint in self.context_data:
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
    
    def rename_voice(self, old_name: str, new_name: str) -> None:
        """
        Renames a voice in the context.

        Arguments:
            old_name (str): The current name of the voice.
            new_name (str): What the voice should be renamed to.
        """
        context = []

        for datapoint in self.get_context_data().data_points:
            if type(datapoint.source) == ContextSource_Voice:
                if datapoint.source.speaker == old_name:
                    dp = ContextDatapoint(
                        source=ContextSource_Voice(
                            speaker=new_name,
                        ),
                        content=datapoint.content
                    )
                    context.append(dp)
                else:
                    context.append(datapoint)

        self._overwrite_context(context=context)

    def _prepare_context_data(self) -> list[dict]:
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
    
    def _take_average_embedding(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(embeddings), dim=0)