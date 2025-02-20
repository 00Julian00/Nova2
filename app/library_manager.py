"""
Description: This script is responsible for interaction with json files designated as libraries which store large amounts of static data.
"""

from pathlib import Path
import json

class LibraryManager:
   def __init__(self) -> None:
       """
       This class is used to interact with libaries.
       """
       self._library_path = Path(__file__).parent.parent / "data" / "libraries"

   def retrieve_datapoint(self, library_name: str, datapoint_name: str) -> dict:
        """
        Retrieve a specific datapoint from a libary.

        Arguments:
            libary_name (str): The name of the libary to search for the datapoint in.
            datapoint_name (str): The name of the datapoint to look for.

        Returns:
            dict: The contents of the datapoint.
        """
        try:
            data = json.loads((self._library_path / f"{library_name}.json").read_text())
            return data[datapoint_name]
        except FileNotFoundError:
            raise FileNotFoundError(f"The library '{library_name}' does not exist.")
        except KeyError:
            raise KeyError(f"The datapoint '{datapoint_name}' does not exist in the library '{library_name}'.")