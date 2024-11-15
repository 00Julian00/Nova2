"""
Description: This script manages the LLM tools.
"""

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
                parameters: list[LLMToolParameter]
                ) -> None:

        """
        Defines a tool that can be used by the LLM.
        """

        self.name = name
        self.description = description
        self.parameters = parameters

class LLMToolCallParameter:
    def __init__(
                self,
                name: str,
                value: str #The value is always a string. Casting needs to be handled by the tool that is executed. Alternativly leave the type ambigous and look up the type in the tool's parameter definition.
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
                parameters: list[LLMToolCallParameter]
                ) -> None:

        """
        Defines a tool call made by the LLM.
        """

        self.name = name
        self.parameters = parameters

class ToolManager:
    def __init__(self) -> None:
        pass
    
    def load_tools(self) -> list[LLMTool]:
        pass
