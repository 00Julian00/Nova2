"""
Description: Implements the tool API.
"""
from Nova2.app.context_manager import ContextManager
from Nova2.app.context_data import ContextDatapoint, ContextSource_ToolResponse
from Nova2.app.api_implementation import NovaAPI as APIImpl

class ToolAPI(APIImpl):
    """
    Primary API to interact with the Nova system.
    """
    def add_to_context(self, name: str, content: str, tool_call_id: str) -> None: # type: ignore
        dp: ContextDatapoint = ContextDatapoint(
            source=ContextSource_ToolResponse(name=name, id=tool_call_id),
            content=content
        )
        ContextManager().add_to_context(datapoint=dp)