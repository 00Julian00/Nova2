"""
Description: A script that holds various helper code.
"""

import sys
import os
from contextlib import contextmanager
import warnings
from functools import wraps

registry = {}

@contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
class Singleton:
    """
    A class that inherits from this class will automatically be a singleton.
    """
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]
    
def deprecated(func=None, *, warning=""):
    """
    Marks a function as deprecated.
    Can be used as @deprecated or @deprecated(warning="custom message")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use standard warning message if none is provided
            warning_msg = f"Function {func.__name__} is deprecated and will be removed in a future update."
            if warning != "":
                warning_msg = warning
            warnings.warn(
                warning_msg,
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    
    # Handle being called as @deprecated or @deprecated(warning="...")
    if func is None:
        # Called with parameters: @deprecated(warning="...")
        return decorator
    # Called without parameters: @deprecated
    return decorator(func)