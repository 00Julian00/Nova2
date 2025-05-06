"""
Description: A script that holds various helper code.
"""

import sys
import os
from contextlib import contextmanager
import inspect
from typing import get_type_hints

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

def overload(func):
    """
    Decorator to allow function overloading with type checking.
    """
    if func.__qualname__ not in registry:
        registry[func.__qualname__] = []
    if func not in registry[func.__qualname__]:
        registry[func.__qualname__].append(func)
    
    def wrapper(*args, **kwargs):
        result = None
        for candidate in registry[func.__qualname__]:
            try:
                # Check for correct number of arguments
                signature = inspect.signature(candidate)
                bound_args = signature.bind(*args, **kwargs)
                
                # Check for correct types
                type_hints = get_type_hints(candidate)
                type_match = True
                
                for param_name, param_value in bound_args.arguments.items():
                    if param_name in type_hints:
                        expected_type = type_hints[param_name]
                        # Check if the argument is of the expected type
                        if not isinstance(param_value, expected_type):
                            type_match = False
                            break
                
                if type_match:
                    if result:
                        raise Exception(f"Multiple overloads of {func.__name__} match the arguments: {args}, {kwargs}")
                    result = candidate
            except TypeError:
                pass
        
        if not result:
            raise Exception(f"No overload matches the arguments: {args}, {kwargs}")
        
        return result(*args, **kwargs)
    
    return wrapper
class Singleton:
    """
    A class that inherits from this class will automatically be a singleton.
    """
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]