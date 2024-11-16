"""
Description: This script is the API for the tools that they can use to interact with Nova and receive data from the system via the event system.
"""

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

    #LLM interaction.
    def add_to_context(self, context: str, tool_name: str) -> None:
        pass