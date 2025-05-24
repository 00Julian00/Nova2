"""
Description: This script manages the LLM tools.
"""

import json
from pathlib import Path
import warnings
import importlib.util
import ast

from Nova2.app.tool_data import LLMTool, LLMToolParameter, LLMToolCall
from Nova2.app.context_manager import ContextManager
from Nova2.app.context_data import ContextDatapoint, ContextSource_System
from Nova2.app.library_manager import LibraryManager
from Nova2.app.helpers import Singleton
from Nova2.app.api_tools import NovaAPI
from Nova2.tool_api.tool_api import ToolBaseClass

class ToolManager(Singleton):
    def __init__(self) -> None:
        """
        This class manages the internal and external tools, as well as their execution.
        """
        self._loaded_tools: list[LLMTool] = []
        self._tool_api_instance = NovaAPI()
    
    def load_tools(self, load_internal: bool = True, **kwargs) -> list[LLMTool]:
        """
        Loads all tools from the tools folder. Also imports all .py files in the tools folder, so that inheritance is possible (importing happens in ExternalToolManager).

        Returns:
            list[LLMTool]: A list of all loaded tools.
        """

        if "include" in kwargs.keys() and "exclude" in kwargs.keys():
            raise Exception("\"include\" and \"exclude\" parameter of \"load_tools\" can not be used at the same time. Use only one or none.")

        tool_list = []
        is_whitelist = False

        if "include" in kwargs.keys():
            tool_list = kwargs["include"]
            is_whitelist = True
        elif "exclude" in kwargs.keys():
            tool_list = kwargs["exclude"]

        internals = LibraryManager().retrieve_datapoint(library_name="internal_tools", datapoint_name="internal_tools")

        # Loads all the tools metadata and creates LLMTool objects from them.
        tools_dir = Path(__file__).parent.parent / "tools"

        for tool_dir in tools_dir.iterdir():
            # Load the metadata
            metadata_path = tool_dir / "metadata.json"

            tool_name = ""

            metadata = None

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    try:
                        metadata = json.load(f)

                        tool_name = metadata["name"]

                        # Check wether this tool should be loaded
                        if not load_internal and tool_name in internals:
                            continue

                        if is_whitelist:
                            if tool_name not in tool_list:
                                continue
                        else:
                            if tool_name in tool_list:
                                continue

                        parameters = []
                        if "parameters" in metadata:
                            for param in metadata["parameters"]:
                                parameters.append(LLMToolParameter(**param))

                    except: # Likely wrong file format. Skip.
                        warnings.warn(f"Error accessing metadata of tool {tool_dir.name}. Skipping.")
                        continue
            else:
                warnings.warn(f"No metadata file found for tool {tool_dir.name}. Skipping.")
                continue

            # Load the scripts into memory and run on_startup() as well as saving the class
            inherited_class: ToolBaseClass = None # type: ignore
            for script in tool_dir.glob("*.py"):
                try:
                    spec = importlib.util.spec_from_file_location(script.stem, script)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        classes = [getattr(module, name) for name in dir(module) if isinstance(getattr(module, name), type)]

                        # Find a class that inherits from the tool base class
                        for cls in classes:
                            if issubclass(cls, ToolBaseClass) and cls != ToolBaseClass:
                                class_instance = cls()

                                if inherited_class != None:
                                    raise Exception(f"More then one class found that inherits from ToolBaseClass in tool {tool_name}. Only one class can inherit from ToolBaseClass.")

                                inherited_class = class_instance

                        # Inject the API into the tool
                        inherited_class.api = self._tool_api_instance
                        inherited_class.on_startup() # Run initialization code

                except Exception as e:
                    warnings.warn(f"Failed to load script {script} into memory: {e}")

                if not inherited_class:
                    continue

                tool = LLMTool(
                    name=tool_name,
                    description=metadata["description"], 
                    parameters=parameters,
                    _instance=inherited_class
                )
                self._loaded_tools.append(tool)

        return self._loaded_tools
    
    # Man I do love sketchy solutions to circular imports
    def _get_base_class(self):
        from Nova2.tool_api.tool_api import ToolBaseClass
        return ToolBaseClass
    
    def _validate_tool_call(self, call: LLMToolCall, tool: LLMTool) -> bool:
        # Verifies wether all paramaters in the call are defined in the tool and wether the call
        # contains all required parameters

        # All parameters defined?
        for param in call.parameters:
            if not any(tool_param.name == param.name for tool_param in tool.parameters):
                return False
        
        # All required parameters present in call?
        for param in tool.parameters:
            if param.required:
                if not any(call_param.name == param.name for call_param in call.parameters):
                    return False
                
        return True
        
    def execute_tool_call(self, tool_calls: list[LLMToolCall]) -> None:
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
                        # Validate the tool call first
                        if not self._validate_tool_call(tool_call, tool):
                            dp = ContextDatapoint(
                                source=ContextSource_System(),
                                content=f"Tool \"{tool_call.name}\" could not be executed because it's call structure is incorrect. Please correct your tool call."
                            )

                            ContextManager().add_to_context(datapoint=dp)

                            return
                        
                        params = {}
                        for param in tool_call.parameters:
                            # Use ast to cast to an apropriate datatype
                            try:
                                casted_param = ast.literal_eval(param.value)
                                params[param.name] = casted_param
                            except:
                                # Parameter failed to be cast so most likely it should remain a string
                                params[param.name] = param.value

                        tool._instance._tool_call_id = tool_call.id
                        tool._instance.on_call(**params)
                        
                        return

                dp = ContextDatapoint(
                    source=ContextSource_System(),
                    content=f"Tool \"{tool_call.name}\" is not loaded in memory and can therefore not be executed."
                )

                ContextManager().add_to_context(datapoint=dp)

                warnings.warn(f"Attempted to call tool {tool_call.name}, but no tool with this name is loaded.")

            except Exception as e:
                dp = ContextDatapoint(
                    source=ContextSource_System(),
                    content=f"Tool \"{tool_call.name}\" failed to execute for an unknown reason."
                )

                ContextManager().add_to_context(datapoint=dp)

                warnings.warn(f"Tool {tool_call.name} failed to execute with error: {e}")