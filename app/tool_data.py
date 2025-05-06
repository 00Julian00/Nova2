"""
Description: Holds all data required for tool use.
"""

from typing import Callable, List
from dataclasses import dataclass

from .shared_types import (
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
    type: str
    required: bool

@dataclass
class LLMTool(LLMToolBase):
    name: str
    description: str
    parameters: List[LLMToolParameter]

    def to_dict(self) -> dict:
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
class LoadedTool(LoadedToolBase):
    name: str
    class_instance: Callable

@dataclass
class LLMToolCallParameter(LLMToolCallParameterBase):
    name: str
    value: str

@dataclass
class LLMToolCall(LLMToolCallBase):
    name: str
    parameters: list[LLMToolCallParameter]
    id: str
