"""
Description: Holds all data required for tool use.
"""

from typing import Callable, List
from dataclasses import dataclass

@dataclass
class LLMToolParameter:
    """
    Defines a parameter for a tool.

    Arguments:
        name (str): The name of the parameter. Should be short and accurate.
        description (str): A description in natural language. Helps the LLM to understand how to use the parameter.
        type (str): What datatype the parameter is, i.e. bool, int, string etc.
        required (bool): Whether the parameter has to be parsed.
    """
    name: str
    description: str
    type: str
    required: bool

@dataclass
class LLMTool:
    """
    Defines a tool that can be used by the LLM.

    Arguments:
        name (str): The name of the tool. Should be short and accurate.
        description (str): A description in natural language. Helps the LLM to understand how to use the tool.
        parameters (List[LLMToolParameter]): A list of parameters the tool can take.
    """
    name: str
    description: str
    parameters: List[LLMToolParameter]

    def to_dict(self) -> dict:
        """
        Converts a list of LLMTools to the proper json format for the LLM and returns it as a dictionary.
        """
        properties = {}
        required_params = []
        
        # Turn the parameters into a single properties object
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.required:
                required_params.append(param.name)
        
        tool = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                }
            }
        }

        return tool

@dataclass
class LoadedTool:
    """
    Defines a list of class instance of a tool together with its name from the metadata.
    """
    name: str
    class_instance: Callable

@dataclass
class LLMToolCallParameter:
    """
    Defines a parameter for a tool call.
    """
    name: str
    value: str

@dataclass
class LLMToolCall:
    """
    Defines a tool call made by the LLM.
    """
    name: str
    parameters: list[LLMToolCallParameter]
    id: str
