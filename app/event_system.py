"""
Description: This script is a C# like event system implementation.
"""

_events: dict[str, list[callable]] = {}

def event_exists_error_handling(func: callable):
    """
    Raises an exception if something tries to interact with an event that does not exist.
    """
    def wrapper(event_name: str, *args, **kwargs):
        if event_name not in _events:
            raise Exception(f"Event {event_name} does not exist.")
        return func(event_name, *args, **kwargs)
    return wrapper

def define_event(event_name: str) -> None:
    """
    Defines a new event.

    Arguments:
        event_name (str): The name of the new event.
    """
    if event_name in _events:
        raise Exception(f"Event {event_name} already exists.")

    _events[event_name] = []

@event_exists_error_handling
def subscribe(event_name: str, callback: callable) -> None:
    """
    Subscribe to an event.

    Arguments:
        event_name (str): The event that should be subscribed to.
        callback (callable): The callable that will be called when the event is triggered.
    """
    _events[event_name].append(callback)

@event_exists_error_handling
def unsubscribe(event_name: str, callback: callable) -> None:
    """
    Unsubscribe from an event.

    Arguments:
        event_name (str): The event that should be unsubscribed from.
        callback (callable): The callable that will no longer be called when the event is triggered.
    """
    _events[event_name].remove(callback)

@event_exists_error_handling
def is_subscribed(event_name: str, callback: callable) -> bool:
    """
    Checks if a callable is subscribed to an event.

    Arguments:
        event_name (str): The event that should be checked.
        callback (callable): The callable that should be checked.

    Returns:
        bool: True if the callable is subscribed to the event.
    """
    return callback in _events[event_name]

@event_exists_error_handling
def trigger_event(event_name: str, *args, **kwargs) -> None:
    """
    Triggers an event. All subscribed callables will be called.

    Arguments:
        event_name (str): The event that should be triggered.
        args, kwargs: Addidtional arguments that should be parsed to the callabales.
    """
    for callback in _events[event_name]:
        try:
            callback(*args, **kwargs)
        except Exception as e:
            print(f"Unable to call {callback} from event {event_name}. Error: {e}")

def event_exists(event_name: str) -> bool:
    """
    Checks if an event exists.

    Arguments:
        event_name (str): Check if an event with this name exists.

    Returns:
        bool: True if the event exists.
    """
    return event_name in _events