"""
Description: This script is responsible for interaction with json files designated as libraries which store large amounts of static data.
"""

import json
import os

class LibraryManager:
    def __init__(self) -> None:
        self._library_path = os.path.join(os.path.dirname(__file__), "..", "data", "libraries")

    def retrieve_datapoint(self, library_name: str, datapoint_name: str) -> dict:
        try:
            with open(os.path.join(self._library_path, f"{library_name}.json"), "r") as file:
                return json.load(file)[datapoint_name]
        except FileNotFoundError:
            raise FileNotFoundError(f"The library '{library_name}' does not exist.")
        except KeyError:
            raise KeyError(f"The datapoint '{datapoint_name}' does not exist in the library '{library_name}'.")