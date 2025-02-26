"""
Description: Holds all classes that are sources for context data.
"""

from typing import List

class ContextSourceBase:
    def __init__(self):
        pass
    
    @classmethod
    def get_all_sources(cls) -> List[type]:
        return cls.__subclasses__()

class Voice(ContextSourceBase):
    def __init__(
                self,
                speaker: str
                ) -> None:
        self.speaker = speaker

class Assistant(ContextSourceBase):
    def __init__(self) -> None:
        pass

class ToolResponse(ContextSourceBase):
    def __init__(
                self,
                name: str,
                id: str
                ) -> None:
        self.name = name
        self.id = id

class System(ContextSourceBase):
    def __init__(self) -> None:
        pass