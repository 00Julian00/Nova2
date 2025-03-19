"""
Description: This script is the API for the tools that they can use to interact with Nova and receive data from the system via the event system.
"""

from typing import List

from app.API import NovaAPI

class Nova(NovaAPI):
    def __init__(self) -> None:
        """
        This class provides a simple API for external tools to interact with internal logic.
        """
        super().__init__()

# Used to run methods inside tools in the "tools" folder via inheritance.
class ToolBaseClass:
    """
    A tool must have a class that inherits from this class, or it can not be used by the system.
    """
    def __init__(self) -> None:
        self._tool_call_id = None

    @classmethod
    def get_subclasses(cls) -> List[type]:
        """
        Returns all subclasses of the current class.

        Returns:
            list[type]: The subclasses of the current class.
        """
        return cls.__subclasses__()
    
    # Custom methods for tools
    def on_startup(self) -> None:
        """
        This method will be called once when the system starts.
        Run initialization logic here.
        """
        pass

    def on_call(self) -> None:
        """
        This method will be called when the tools is executed. Collect parameters and start tool logic here.
        """
        pass