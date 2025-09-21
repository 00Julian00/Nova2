from Nova2.tool_api import ToolBaseClass

from Nova2.app import api_tools
from Nova2.app import database_manager

class Main(ToolBaseClass):
    def on_startup(self):
        self._db = database_manager.MemoryEmbeddingDatabaseManager()
        self._api = api_tools.ToolAPI()

    @ToolBaseClass.tool
    def memorize(self, new_memory: str) -> None:
        """
        Saves a new long-term memory to the database that can be retrieved later.

        Arguments:
            new_memory (str): The memory to be saved.
        """
        self._db.create_new_entry(text=new_memory)

        self._api.add_to_context("Save memory", "Memory was saved to the database.", tool_call_id=self._tool_call_id) # type: ignore