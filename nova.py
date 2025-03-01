"""
Description: Acts as the API for the entire Nova system.
"""

from app.API import *

class Nova(NovaAPI):
    def __init__(self) -> None:
        """
        API to easily construct an AI assistant using the Nova framework.
        """
        super().__init__()

    def add_to_context(self):
        raise NotImplementedError()