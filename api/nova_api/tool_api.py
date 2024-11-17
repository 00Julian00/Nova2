"""
Description: This script is the API for the tools that they can use to interact with Nova and receive data from the system via the event system.
"""

from pathlib import Path
import importlib.util

class Nova:
    def __init__(self) -> None:
        pass
    
    #Event system.
    def subscribe_to_event(self, event_name: str, callback: callable) -> None:
        pass

    def unsubscribe_from_event(self, event_name: str, callback: callable) -> None:
        pass

    def is_subscribed_to_event(self, event_name: str, callback: callable) -> bool:
        pass

    def event_exists(self, event_name: str) -> bool:
        pass

    #LLM interaction.
    def add_to_context(self, context: str, tool_name: str) -> None:
        pass

class ToolBaseClass: #Used to run functions inside tools in the "tools" folder.
    def __init__(self) -> None:
        pass

    @classmethod
    def get_subclasses(cls) -> list[type]:
        """
        Returns all subclasses of the current class.
        """

        return cls.__subclasses__()
    
    #Custom functions for tools.
    def on_startup(cls) -> None:
        """
        This function will be called once when the system starts.
        Subscribe to events and run other initialization logic here.
        """
        pass

class ExternalToolManager:
    """
    Manages the tools in the "tools" folder. Also manages tool initialization and inheritance. Is controled from "ToolManager".
    """
    
    def __init__(self) -> None:
        pass

    def initialize_tools(self) -> None:
        """
        Initializes all tools.
        """

        #Load all .py files in the tools folder into memory.
        tools_dir = Path(__file__).parent.parent.parent / "tools"
        for tool_dir in tools_dir.iterdir():
            if tool_dir.is_dir():
                for py_file in tool_dir.glob("*.py"):
                    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

        #Run the on_startup function for all tool classes.
        for tool_class in ToolBaseClass.get_subclasses():
            tool_class.on_startup(self)
