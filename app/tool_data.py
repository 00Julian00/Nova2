"""
Description: Holds all data required for tool use.
"""

from typing import Callable, List

class LLMToolParameter:
    def __init__(
                self,
                name: str,
                description: str,
                type: str,
                required: bool
                ) -> None:
        """
        Defines a parameter for a tool.

        Arguments:
            name (str): The name of the parameter. Should be short and accurate.
            description (str): A description in natural language. Helps the LLM to understand how to use the parameter.
            type (str): What datatype the parameter is, i.e. bool, int, string etc.
            required (bool): Wether the parameter has to be parsed.
        """
        self.name = name
        self.description = description
        self.type = type
        self.required = required

class LLMTool:
    def __init__(
                self,
                name: str,
                description: str,
                parameters: List[LLMToolParameter]
                ) -> None:
        """
        Defines a tool that can be used by the LLM.

        Arguments:
            name (str): The name of the tool. Should be short and accurate.
            description (str): A description in natural language. Helps the LLM to understand how to use the tool.
            parameters (List[LLMToolParameter]): A list of parameters the tool can take.
        """
        self.name = name
        self.description = description
        self.parameters = parameters

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

class LoadedTool:
    def __init__(
                self,
                name: str,
                class_instance: Callable,
                ) -> None:
        """
        Defines a list of class instance of a tool together with its name from the metadata.
        """
        self.name = name
        self.class_instance = class_instance

class LLMToolCallParameter:
    def __init__(
                self,
                name: str,
                value: str # The value is always a string. Casting needs to be handled by the tool that is executed. Alternativly leave the type ambigous and look up the type in the tool's parameter definition.
                ) -> None:
        """
        Defines a parameter for a tool call.
        """
        self.name = name
        self.value = value

class LLMToolCall:
    def __init__(
                self,
                name: str,
                parameters: list[LLMToolCallParameter],
                id: str
                ) -> None:
        """
        Defines a tool call made by the LLM.
        """
        self.name = name
        self.parameters = parameters
        self.id = id