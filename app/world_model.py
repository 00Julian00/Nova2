#!This script is in the experiemntal stage is not used currently
"""
Description: This script is responsible for creating and managing a world model which is a set of information weighted bei their importance to the current situation.

How it works: The system is based on an entity database. This stores not only entities the system knows about and information about them, but also weighted connections between entities.
If an entity is mentioned in current context, the information about that entity will be assigned the weight of 0.5. The entities connected to that entities will also be introduced into the world model with
a weight of the base entity multiplied by the connection weight. The weight of any given entity will degrade overtime if they are no longer part of the conversation and the relevance of their information will therefore
also decrease.
"""

class Entity: #An entity is a person, object, or any sort of concept that can be expressed as a noun, like a company.
    def __init__(self) -> None:
        self.db_index: str
        self.relevance: float

class WorldModel:
    def __init__(self) -> None:
        self.entities: list[Entity]