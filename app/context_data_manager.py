"""
Description: This script manages the context.json file. It provides an interface to add various data to the context.
"""

import json
from typing import List
import os
from datetime import datetime

class ContextDataManager:
    def __init__(self) -> None:
        self._context_file = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'data'), 'context.json')
        self._context_data = self._prepare_context_data()
    
    def add_spoken_sentence_to_context(self, sentence: str, voice_name: str) -> None:
        """
        Adds the spoken sentence to the context.json file.
        """
        self._context_data.append(
            {
                "source": {
                    "voice": voice_name
                },
                "content": sentence,
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

    def get_context_data(self) -> List[dict]:
        return self._context_data

    def _prepare_context_data(self) -> List:
        """
        Creates a context.json file if it does not exist.
        Returns the content of the context.json file.
        """
        if not os.path.exists(self._context_file):
            with open(self._context_file, 'w') as file:
                json.dump([], file)

        with open(self._context_file, 'r') as file:
            context_data = json.load(file)

        return context_data