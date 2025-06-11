"""
Description: Acts as the API for the entire Nova system.
"""

from Nova2.app.api_implementation import *
from Nova2.app.recovery_manager import automated_recovery

class Nova(NovaAPI):
    def __init__(self) -> None:
        """
        API to easily construct an AI assistant using the Nova framework.
        """
        super().__init__()