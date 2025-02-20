"""
Description: This script is the API for the tools that they can use to interact with Nova and receive data from the system via the event system.
"""

from app import event_system
from app import context_data_manager

class Nova:
    def __init__(self) -> None:
        """
        This class provides a simple API for external tools to interact with internal logic.
        """
        pass
    
    # Event system
    def subscribe_to_event(self, event_name: str, callback: callable) -> None:
        event_system.subscribe(event_name, callback)

    def unsubscribe_from_event(self, event_name: str, callback: callable) -> None:
        event_system.unsubscribe(event_name, callback)

    def is_subscribed(self, event_name: str, callback: callable) -> bool:
        return event_system.is_subscribed(event_name, callback)

    def event_exists(self, event_name: str) -> bool:
        return event_system.event_exists(event_name)

    # LLM interaction
    @DeprecationWarning
    def add_to_context(self, context: str, tool_name: str) -> None:
        context_data_manager.add_to_context(source={"tool": tool_name}, content=context)

# Used to run methods inside tools in the "tools" folder via inheritance.
class ToolBaseClass:
    """
    A tool must have a class that inherits from this class, or it can not be used by the system.
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def get_subclasses(cls) -> list[type]:
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
        Subscribe to events and run other initialization logic here.
        """
        pass

    def on_call(self, **kwargs) -> None:
        """
        This method will be called when the tools is executed. Collect parameters and start tool logic here.
        """
        pass