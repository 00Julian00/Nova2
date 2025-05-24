from Nova2.tool_api import *

from Nova2.app import database_manager

class Main(ToolBaseClass):
    def on_startup(self):
        #self._db = database_manager.MemoryEmbeddingDatabaseManager()
        pass

    def on_call(self, **kwargs):
        memory = kwargs["New memory"]
        self._db.create_new_entry(text=memory)

        self._api.add_to_context("Save memory", "Memory was saved to the database.", self._tool_call_id)