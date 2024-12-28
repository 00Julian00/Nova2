"""
Description: Holds all data related to context data.
"""

from datetime import datetime

class ContextDatapoint:
    def __init__(
            self,
            source, # All possible sources are in context_sources.py
            content: str,
            timestamp: datetime
            ) -> None:
        """
        This class holds a singular datapoint in the context.
        """
        self.source = source
        self.content = content
        self.timestamp = timestamp

class Context:
    def __init__(
            self,
            data_points: list[ContextDatapoint]
            ) -> None:
        """
        This class stores context which is a list of datapoints all with source, content and timestamp.
        """
        self.data_points = data_points