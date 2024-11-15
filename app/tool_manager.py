"""
Description: This script manages the LLM tools.
"""

import json
from pathlib import Path

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
        """
        Loads all tools from the tools folder.
        """

        tools = []
        tools_dir = Path(__file__).parent.parent / "tools"
        
        for tool_dir in tools_dir.iterdir():
            manifest_path = tool_dir / "manifest.json"
            
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    try:
                        manifest_list = json.load(f)

                        for manifest in manifest_list:
                            parameters = []
                            if "parameters" in manifest:
                                for param in manifest["parameters"]:
                                    parameters.append(LLMToolParameter(**param))

                            tool = LLMTool(
                                        name=manifest["name"],
                                        description=manifest["description"], 
                                        parameters=parameters
                            )
                            tools.append(tool)
                    except: #Likely wrong file format. Skip.
                        continue
        
        return tools
    
    def debug_visualize_loaded_tools(self) -> None:
        """
        Debug function that loads tools and prints their contents.
        """
        
        tools = self.load_tools()
        
        print("\nLoaded Tools:")
        print("-" * 50)
        
        for tool in tools:
            print(f"\nTool: {tool.name}")
            print(f"Description: {tool.description}")
            
            if tool.parameters:
                print("\nParameters:")
                for param in tool.parameters:
                    print(f"  - {param.name}: {param.description}")
                    print(f"    Type: {param.type}")
            else:
                print("\nNo parameters defined")
            
            print("-" * 50)
