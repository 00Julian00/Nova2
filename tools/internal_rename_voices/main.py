from tool_api import ToolBaseClass, Nova

from app import database_manager

class Main(ToolBaseClass):
    def on_startup(self):
        self._db = database_manager.VoiceDatabaseManager()

        self._api = Nova()

    def on_call(self, **kwargs):
        current_name = kwargs["Current voice name"]
        new_name = kwargs["New voice name"]

        self._db.open()
        if not self._db.edit_voice_name(current_name, new_name):
            self._api.add_to_context(name="Rename voice", content=f"Voice {current_name} does not exist in the database.", id=self._tool_call_id)
        else:
            self._api.add_to_context(name="Rename voice", content=f"{current_name} was renamed to {new_name}.", id=self._tool_call_id)