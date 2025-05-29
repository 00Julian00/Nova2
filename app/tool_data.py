"""
Description: Holds all data required for tool use.
"""

from dataclasses import dataclass

from Nova2.tool_api.tool_api import ToolBaseClass
from Nova2.app.interfaces import (
    LLMToolBase,
    LLMToolParameterBase,
    LoadedToolBase,
    LLMToolCallBase,
    LLMToolCallParameterBase
)

@dataclass
class LLMToolParameter(LLMToolParameterBase):
    name: str
    description: str
    datatype: str
    required: bool

@dataclass
class LLMTool(LLMToolBase):
    name: str
    description: str
    parameters: list[LLMToolParameter]
    _instance: callable # type: ignore

    def to_dict(self) -> dict:
        properties = {}
        required_params = []
        
        # Turn the parameters into a single properties object
        for param in self.parameters:
            properties[param.name] = {
                "type": param.datatype,
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
class LoadedTool(LoadedToolBase):
    name: str
    class_instance: ToolBaseClass

@dataclass
class LLMToolCallParameter(LLMToolCallParameterBase):
    """
    Defines a parameter for a tool call.
    """
    name: str
    value: str

@dataclass
class LLMToolCall(LLMToolCallBase):
    """
    Defines a tool call made by the LLM.
    """
    name: str
    parameters: list[LLMToolCallParameter]
    id: str
