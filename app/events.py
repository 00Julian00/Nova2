"""
This script is a C# like event system implementation.
"""

_events: dict[str, list[callable]] = {}

def define_event(event_name: str) -> None:
    """
    Defines an event.
    """

    if event_name not in _events:
        _events[event_name] = []
    else:
        print(f"Event {event_name} already exists.")

def event_exists(func):
    def wrapper(event_name: str, *args, **kwargs):
        if event_name not in _events:
            print(f"Event {event_name} does not exist.")
            return
        return func(event_name, *args, **kwargs)
    return wrapper

@event_exists
def subscribe(event_name: str, callback: callable) -> None:
    """
    Subscribes a callback to an event.
    """

    if callback not in _events[event_name]:
        _events[event_name].append(callback)
    else:
        print(f"Callback {callback} already subscribed to event {event_name}.")

@event_exists
def unsubscribe(event_name: str, callback: callable) -> None:
    """
    Unsubscribes a callback from an event.
    """
    
    if callback in _events[event_name]:
        _events[event_name].remove(callback)
    else:
        print(f"Callback {callback} not found in event {event_name}.")

@event_exists
def is_subscribed(event_name: str, callback: callable) -> bool | None:
    """
    Checks if a callback is subscribed to an event. Returns None if the event does not exist.
    """

    return callback in _events[event_name]

@event_exists
def trigger_event(event_name: str, *args, **kwargs) -> None:
    """
    Triggers an event.
    """
    
    for callback in _events[event_name]:
        try:
            callback(*args, **kwargs)
        except Exception as e:
            print(f"Unable to call {callback} from event {event_name}. Error: {e}")