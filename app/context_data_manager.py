"""
Description: This script manages the context.json file. It provides an interface to add various data to the context.
"""

import json
from typing import List
from datetime import datetime
from pathlib import Path

class ContextDataManager:
    def __init__(self) -> None:
        """
        This class is responsible for managing the 'context.json' file in 'data'.
        This includes reading from and writing to it.
        """
        self._context_file = Path(__file__).parent.parent / "data" / "context.json"
        self._context_data = self._prepare_context_data()
    
    def add_to_context(self, source: dict, content: str) -> None:
        """
        Adds content to the context.json file.

        Arguments:
            source (dict): The source of the information.
            content (string): The information itself.
        """
        self._context_data.append(
            {
                "source": source,
                "content": content,
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            }
        )

        self.save_context_data()

    def save_context_data(self) -> None:
        """
        Saves the context data to the context.json file.
        """
        with open(self._context_file, 'w') as file:
            json.dump(self._context_data, file, indent=4)

    def get_context_data(self) -> list[dict]:
        """
        Returns the current context data stored in memory.

        Returns:
            list[dict]: The context data stored in memory.
        """
        return self._context_data

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