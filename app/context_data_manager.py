"""
Description: This script manages the context.json file. It provides an interface to add various data to the context.
"""

import json
from typing import List
from pathlib import Path

from .context_data import ContextDatapoint
from .context_sources import ContextSourceBase

class ContextDataManager:
    def __init__(self) -> None:
        """
        This class is responsible for managing the 'context.json' file in 'data'.
        This includes reading from and writing to it.
        """
        self._context_file = Path(__file__).parent.parent / "data" / "context.json"
        self._context_data = self._prepare_context_data()
    
    def add_to_context(self, datapoint: ContextDatapoint) -> None:
        """
        Adds content to the context.json file.

        Arguments:
            datapoint (ContextDatapoint): The datapoint that will be added to the context.
        """
        self._context_data.append(datapoint.format())

        self.save_context_data()

    def save_context_data(self) -> None:
        """
        Saves the context data to the context.json file.
        """
        with open(self._context_file, 'w') as file:
            json.dump(self._context_data, file, indent=4)

    def get_context_data(self) -> List[ContextDatapoint]:
        """
        Reconstructs a list of ContextDatapoints from the stored context.

        Returns:
            list[ContextDatapoint]: The context data stored in memory.
        """
        sources = ContextSourceBase.get_all_sources()

        datapoints = []

        for datapoint in self._context_data:
            source_instance = None
            # Find the correct source
            for source in sources:
                if source.__name__ == datapoint["source"]["type"] and source.__name__ != "ContextSourceBase":
                    metadata = datapoint["source"]["metadata"]
                    source_instance = source(**metadata)
                    break
            
            if not source_instance:
                raise Exception(f"Got unknown context source {datapoint["source"]["type"]}")

            datapoint_instance = ContextDatapoint(
                source=source_instance,
                content=datapoint["content"]
                )
            
            datapoint_instance.timestamp = datapoint["timestamp"]
            datapoints.append(datapoint_instance)

        return datapoints

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