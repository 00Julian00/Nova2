"""
Description: This script is the API for the tools that they can use to interact with Nova and receive data from the system via the event system.
"""

from Nova2.app.api_base import APIAbstract

# Used to run methods inside tools in the "tools" folder via inheritance.
class ToolBaseClass:
    """
    A tool must have a class that inherits from this class, or it can not be used by the system.
    """
    def __init__(self) -> None:
        self._tool_call_id = None
        self.api: APIAbstract = None # type: ignore

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        disallowed_overrides = [
            "__init__",
            "__get_subclasses__",
            "tool",
            "__get_tools__"
        ]
        
        for method in disallowed_overrides:
            if method in cls.__dict__:
                raise TypeError(f"Tool class {cls.__name__} cannot override the method '{method}'. This is reserved for tool API functionality.")

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
        Run initialization logic here.
        """
        pass

    def on_call(self) -> None:
        """
        This method will be called when the tools is executed. Collect parameters and start tool logic here.
        """
        pass

    @classmethod
    def tool(cls, func):
        """
        Marks a function as a tool method.
        """
        func.__is_tool__ = True
        return func
    
    def __get_tools__(self) -> list[callable]: # type: ignore
        """
        Returns all methods that are marked as tools.
        
        Returns:
            list[callable]: The tool methods.
        """
        return [
            getattr(self, name)
            for name in dir(self)
            if callable(getattr(self, name))
            and hasattr(getattr(self, name), "__is_tool__")
            and getattr(self, name).__is_tool__
        ]