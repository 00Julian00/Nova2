"""
Description: This script is the API for the tools that they can use to interact with Nova and receive data from the system via the event system.
"""

from pathlib import Path
import importlib.util
from app import event_system
from app import context_data_manager

class Nova:
    def __init__(self) -> None:
        pass
    
    #Event system.
    def subscribe_to_event(self, event_name: str, callback: callable) -> None:
        event_system.subscribe(event_name, callback)

    def unsubscribe_from_event(self, event_name: str, callback: callable) -> None:
        event_system.unsubscribe(event_name, callback)

    def is_subscribed(self, event_name: str, callback: callable) -> bool:
        return event_system.is_subscribed(event_name, callback)

    def event_exists(self, event_name: str) -> bool:
        return event_system.event_exists(event_name)

    #LLM interaction.
    def add_to_context(self, context: str, tool_name: str) -> None:
        context_data_manager.add_to_context(source={"tool": tool_name}, content=context)

class ToolBaseClass: #Used to run methods inside tools in the "tools" folder via inheritance.
    def __init__(self) -> None:
        pass

    @classmethod
    def get_subclasses(cls) -> list[type]:
        """
        Returns all subclasses of the current class.
        """

        return cls.__subclasses__()
    
    #Custom methods for tools.
    def on_startup(self) -> None:
        """
        This method will be called once when the system starts.
        Subscribe to events and run other initialization logic here.
        """
        pass

class ExternalToolManager:
    """
    Manages the tools in the "tools" folder. Also manages tool initialization and inheritance. Is controled from "ToolManager".
    """
    
    def __init__(self) -> None:
        self._tools = []

    def initialize_tools(self) -> None:
        """
        Initializes all tools by loading them into memory and calling their startup methods.
        """

        #Load all .py files in the tools folder into memory, else they can't be used via inheritance or event system.
        tools_dir = Path(__file__).parent.parent.parent / "tools"
        for tool_dir in tools_dir.iterdir():
            if tool_dir.is_dir():
                for py_file in tool_dir.glob("*.py"):
                    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

        #Run the on_startup method for all tool classes.
        for tool_class in ToolBaseClass.get_subclasses():
            tool_instance = tool_class()
            self._tools.append(tool_instance)
            tool_instance.on_startup()
