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

    def add_to_context(self, source: ContextSourceBase, content: str):
        dp = ContextDatapoint(
            source=source,
            content=content
        )
        ContextManager().add_to_context(datapoint=dp)

    def add_datapoint_to_context(self, datapoint: ContextDatapoint):
        ContextManager().add_to_context(datapoint=datapoint)