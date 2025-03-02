from tool_api import ToolBaseClass, Nova

from app import database_manager

class Main(ToolBaseClass):
    def on_startup(self):
        self._db = database_manager.MemoryEmbeddingDatabaseManager()

        self._api = Nova()

    def on_call(self, **kwargs):
        memory = kwargs["New memory"]
        self._db.open()
        self._db.create_new_entry(text=memory)
        self._db.close()

        self._api.add_to_context("Save memory", "Memory was saved to the database.", self._tool_call_id)