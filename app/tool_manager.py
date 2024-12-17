"""
Description: This script manages the LLM tools.
"""

import json
from pathlib import Path

from nova_api.tool_api import ExternalToolManager
from .tool_data import LLMTool, LLMToolParameter

class ToolManager:
    def __init__(self) -> None:
        pass
    
    def load_tools(self) -> list[LLMTool]:
        """
        Loads all tools from the tools folder. Also imports all .py files in the tools folder, so that inheritance is possible (importing happens in ExternalToolManager).
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