from Nova2.tool_api import ToolBaseClass

from Nova2.app import database_manager, context_manager
from Nova2.app import api_tools

class Main(ToolBaseClass):
    def on_startup(self):
        self._db = database_manager.VoiceDatabaseManager()
        self._api = api_tools.ToolAPI()
        self._context = context_manager.ContextManager()

    @ToolBaseClass.tool
    def rename_voice(self, current_name: str, new_name: str) -> None:
        """
        By default, a new voice is called UnknownVoiceX. When the name of the speaker becomes known,
        this tool can be used to rename the voice in the context and the database.

        Arguments:
            current_name (str): The current name of the voice.
            new_name (str): What the voice should be renamed to.
        """
        if not self._db.edit_voice_name(current_name, new_name):
            self._api.add_to_context(name="Rename voice", content=f"Voice {current_name} does not exist in the database.", tool_call_id=self._tool_call_id) # type: ignore
        else:
            self._api.add_to_context(name="Rename voice", content=f"{current_name} was renamed to {new_name}.", tool_call_id=self._tool_call_id) # type: ignore

        self._context.rename_voice(old_name=current_name, new_name=new_name)