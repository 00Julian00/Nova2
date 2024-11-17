"""
This script is a C# like event system implementation.
"""

_events: dict[str, list[callable]] = {}

def event_exists_error_handling(func):
    def wrapper(event_name: str, *args, **kwargs):
        if event_name not in _events:
            raise Exception(f"Event {event_name} does not exist.")
        return func(event_name, *args, **kwargs)
    return wrapper

def define_event(event_name: str) -> None:
    """
    Defines an event.
    """

    if event_name in _events:
        raise Exception(f"Event {event_name} already exists.")

    _events[event_name] = []

@event_exists_error_handling
def subscribe(event_name: str, callback: callable) -> None:
    """
    Subscribes a callback to an event.
    """

    _events[event_name].append(callback)

@event_exists_error_handling
def unsubscribe(event_name: str, callback: callable) -> None:
    """
    Unsubscribes a callback from an event.
    """
    
    _events[event_name].remove(callback)

@event_exists_error_handling
def is_subscribed(event_name: str, callback: callable) -> bool | None:
    """
    Checks if a callback is subscribed to an event. Returns None if the event does not exist.
    """

    return callback in _events[event_name]

@event_exists_error_handling
def trigger_event(event_name: str, *args, **kwargs) -> None:
    """
    Triggers an event.
    """
    
    for callback in _events[event_name]:
        try:
            callback(*args, **kwargs)
        except Exception as e:
            print(f"Unable to call {callback} from event {event_name}. Error: {e}")

def event_exists(event_name: str) -> bool:
    """
    Checks if an event exists.
    """

    return event_name in _events
