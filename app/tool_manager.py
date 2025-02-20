"""
Description: This script manages the LLM tools.
"""

import json
from pathlib import Path
from typing import List
import warnings
import importlib.util
import ast

from tool_api import ToolBaseClass
from .tool_data import LLMTool, LLMToolParameter, LLMToolCall, LoadedTool

class ToolManager:
    def __init__(self) -> None:
        """
        This class manages the internal and external tools, as well as their execution.
        """
        self._loaded_tools = []
    
    def load_tools(self) -> list[LLMTool]:
        """
        Loads all tools from the tools folder. Also imports all .py files in the tools folder, so that inheritance is possible (importing happens in ExternalToolManager).

        Returns:
            list[LLMTool]: A list of all loaded tools.
        """
        # Loads all the tools metadata and creates LLMTool objects from them.
        tools = []
        tools_dir = Path(__file__).parent.parent / "tools"

        for tool_dir in tools_dir.iterdir():
            # Load the metadata
            metadata_path = tool_dir / "metadata.json"

            tool_name = ""

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    try:
                        metadata = json.load(f)

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

                        tool_name = metadata["name"]

                    except: # Likely wrong file format. Skip.
                        warnings.warn(f"Error accessing metadata of tool {tool_dir.name}. Skipping.")
                        continue

            # Load the scripts into memory and run on_startup() as well as saving the class
            inherited_class = None # The class that inherits from ToolBaseClass
            for script in tool_dir.glob("*.py"):
                try:
                    spec = importlib.util.spec_from_file_location(script.stem, script)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        classes = [getattr(module, name) for name in dir(module) if isinstance(getattr(module, name), type)]

                        for cls in classes:
                            if issubclass(cls, ToolBaseClass) and cls != ToolBaseClass:
                                class_instance = cls()
                                class_instance.on_startup() # Run initialization code

                                if inherited_class != None:
                                    raise Exception(f"More then one class found that inherits from ToolBaseClass in tool {tool_name}. Only one class can inherit from ToolBaseClass.")

                                inherited_class = class_instance
                except:
                    warnings.warn(f"Failed to load script {script} into memory.")

            self._loaded_tools.append(LoadedTool(name=tool_name, class_instance=inherited_class))

        return tools
    
    def execute_tool_call(self, tool_calls: List[LLMToolCall]) -> None:
        """
        Attempts to run the scripts associated with the called tools.
        If the tool fails to execute, a warning will be printed, but the functionality of the
        core system will not be affected.

        Arguments:
            tool_calls (List[LLMToolCall]): The tools that should be executed.
        """
        for tool_call in tool_calls:
            try:
                for tool in self._loaded_tools:
                    if tool.name == tool_call.name:
                        params = {}
                        for param in tool_call.parameters:
                            # Use ast to cast to an apropriate datatype
                            try:
                                casted_param = ast.literal_eval(param.value)
                                params[param.name] = casted_param
                            except:
                                # Parameter failed to be cast so most likely it should remain a string
                                params[param.name] = param.value

                        tool.class_instance.on_call(**params)
                        
                        break

                warnings.warn(f"Attempted to call tool {tool_call.name}, but no tool with this name is loaded.")

            except Exception as e:
                warnings.warn(f"Tool {tool_call.name} failed to execute with error: {e}")