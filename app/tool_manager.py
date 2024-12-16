"""
Description: This script manages the LLM tools.
"""

import json
from pathlib import Path

from nova_api.tool_api import ExternalToolManager

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
        Loads all tools from the tools folder. Also imports all .py files in the tools folder, so that inheritance is possible.
        """

        self.tool_api_manager = ExternalToolManager()
        self.tool_api_manager.initialize_tools()

        #Loads all the tools metadata and creates LLMTool objects from them.
        tools = []
        tools_dir = Path(__file__).parent.parent / "tools"
        
        for tool_dir in tools_dir.iterdir():
            metadata_path = tool_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    try:
                        metadata_list = json.load(f)

                        for metadata in metadata_list:
                            parameters = []
                            if "parameters" in metadata:
                                for param in metadata["parameters"]:
                                    parameters.append(LLMToolParameter(**param))

                            tool = LLMTool(
                                        name=metadata["name"],
                                        description=metadata["description"], 
                                        parameters=parameters
                            )
                            tools.append(tool)
                    except: #Likely wrong file format. Skip.
                        print(f"Error loading tool {tool_dir.name}. Skipping.")
                        continue
        
        return tools
    
    def convert_tool_list_to_json(self, tools: list[LLMTool]) -> list[dict]:
        """
        Converts a list of LLMTools to the proper json format for the LLM.
        """

        tools_json = []
        parameters = []
        required_params = []

        #Turn the parameters into json in the format expected by the LLM.
        for tool in tools:
            for param in tool.parameters:
                parameters.append({
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type,
                            "description": param.description
                        }
                    }
                })
                if param.required:
                    required_params.append(param.name)

        tools_json.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "required": required_params
            }
        })

        return tools_json
        
    
    #*Debug function. Can be removed when tools are working.
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