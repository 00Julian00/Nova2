"""
Description: Manages the databases and provides a simple interface
"""

from typing import List, Tuple
import os
import uuid

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client.http import models
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .helpers import hash_embedding

#Database setup
db_people_folder = os.path.join(os.path.dirname(__file__), '..', 'db', 'db_people')
os.makedirs(db_people_folder, exist_ok=True)
db_people_location = os.path.join(db_people_folder, 'db_people.db')
db_people_engine = create_engine(f"sqlite:///{db_people_location}", echo=False)

db_memories_folder = os.path.join(os.path.dirname(__file__), '..', 'db', 'db_memories')
os.makedirs(db_memories_folder, exist_ok=True)
db_memories_location = os.path.join(db_memories_folder, 'db_memories.db')
db_memories_engine = create_engine(f"sqlite:///{db_memories_location}", echo=False)

db_secrets_folder = os.path.join(os.path.dirname(__file__), '..', 'db', 'db_secrets')
os.makedirs(db_secrets_folder, exist_ok=True)
db_secrets_location = os.path.join(db_secrets_folder, 'db_secrets.db')
db_secrets_engine = create_engine(f"sqlite:///{db_secrets_location}", echo=False)

base = declarative_base()

class VoiceDatabaseManager:
    def __init__(self) -> None:
        self._prepare_database()
    
    def create_voice(self, embedding: torch.FloatTensor, name: str) -> None:
        """
        Creates a voice embedding in the Qdrant database.
        """
        self._qdrant_client.upsert(
            collection_name="voice_embeddings",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=self._convert_to_qdrant_format(embedding),
                    payload={"name": name, "hash": hash_embedding(embedding)}
                )
            ]
        )

    def create_unknown_voice(self, embedding: torch.FloatTensor) -> str:
        """
        Creates a new voice with the name "UnknownVoiceX", where X is a number starting from 0. These can later be replaced with the correct name, after the system has obtained the name.
        """
        unknown_counter = 0
        while self.does_voice_exist(f"UnknownVoice{unknown_counter}"):
            unknown_counter += 1

        self.create_voice(embedding, f"UnknownVoice{unknown_counter}")

        return f"UnknownVoice{unknown_counter}"

    def get_voice_from_tensor(self, embedding: torch.FloatTensor) -> Tuple[str, float] | None:
        """
        Searches for the closest voice embedding to the given embedding.
        Returns the name of the closest voice embedding together with the confidence score.	
        """
        search_results = self._qdrant_client.search(
            collection_name="voice_embeddings",
            query_vector=self._convert_to_qdrant_format(embedding),
            limit=1
        )

        if len(search_results) > 0:
            return search_results[0].payload["name"], search_results[0].score
        else:
            return None
        
    def does_voice_exist(self, name: str) -> bool:
        """
        Checks if a voice embedding with the given name exists in the Qdrant database.
        """
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="name",
                    match=models.MatchValue(value=name)
                )
            ]
        )

        search_result = self._qdrant_client.scroll(
            collection_name="voice_embeddings",
            scroll_filter=filter_condition,
            limit=1
        )

        return len(search_result[0]) > 0

    def get_voice_id(self, embedding: torch.FloatTensor) -> int:
        """
        Searches for the ID of a voice embedding in the Qdrant database.
        """
        search_results = self._qdrant_client.search(
            collection_name="voice_embeddings",
            query_vector=self._convert_to_qdrant_format(embedding),
            limit=1
        )

        if len(search_results) > 0:
            return search_results[0].id
        else:
            return None

    #!Incomplete and unused.
    def edit_voice_name(self, embedding: torch.FloatTensor, name: str) -> None:
        """
        INCOMPLETE. DO NOT USE.
        Edits the name of a voice embedding in the Qdrant database.
        """
        voice_id = self.get_voice_id(embedding)

        if voice_id is not None:
            self._qdrant_client.set_payload(
                collection_name="voice_embeddings",
                payload={"name": name},
                points=[voice_id]
            )

    def _prepare_database(self) -> None:
        db_location = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'db'), 'db_embeddings')

        self._qdrant_client = QdrantClient(path=db_location)

        if not self._qdrant_client.collection_exists("voice_embeddings"):
            self._qdrant_client.create_collection(collection_name="voice_embeddings", vectors_config=VectorParams(size=512, distance=Distance.COSINE))

    @staticmethod
    def _convert_to_qdrant_format(embedding: torch.FloatTensor) -> List[float]:
        return embedding.squeeze().cpu().numpy().tolist()

class PeopleDatabaseManager:
    def __init__(self) -> None:
        self._prepare_database()

    def add_person(self, name: str, hash: str, information: List[str]) -> str | None:
        try:
            new_person = Person(name=name, hash=hash, information=information)
            self._session.add(new_person)
            self._session.commit()
        except:
            self._session.rollback()
            return "Error when writing to database."

    def _prepare_database(self) -> None:
        base.metadata.create_all(db_people_engine)
        self._session_factory = sessionmaker(bind=db_people_engine)

        self._session = self._session_factory()

    """def __del__(self) -> None:
        if self._session != None:
            self._session.close()"""

class MemoryDatabaseManager:
    def __init__(self) -> None:
        self._prepare_database()

    def add_memory(self, tag: str, content: List) -> str | None:
        try:
            new_memory = Memory(tag=tag, content=content)
            self._session.add(new_memory)
            self._session.commit()
        except:
            self._session.rollback()
            return "Error when writing to database."

    def _prepare_database(self) -> None:
        base.metadata.create_all(db_memories_engine)
        self._session_factory = sessionmaker(bind=db_memories_engine)

        self._session = self._session_factory()

    """def __del__(self) -> None:
        if self._session != None:
            self._session.close()"""

class SecretsDatabaseManager:
    def __init__(self):
        self._prepare_database()

    def add_secret(self, name: str, encrypted_key: str) -> None:
        try:
            new_secret = Secret(name=name, encrypted_key=encrypted_key)
            self._session.add(new_secret)
            self._session.commit()
        except:
            self._session.rollback()
            raise Exception("Error when writing to database.")
        
    def get_secret(self, name: str) -> str | None:
        secret = self._session.query(Secret).filter_by(name=name).first()

        if secret:
            return secret.encrypted_key
        else:
            return None
        
    def edit_secret(self, name: str, encrypted_key: str) -> None:
        secret = self._session.query(Secret).filter_by(name=name).first()

        try:
            if secret:
                secret.encrypted_key = encrypted_key
                self._session.commit()
        except:
            self._session.rollback()
            raise Exception("Error when writing to database.")

    def _prepare_database(self) -> None:
        base.metadata.create_all(db_secrets_engine)
        self._session_factory = sessionmaker(bind=db_secrets_engine)

        self._session = self._session_factory()

    """def __del__(self) -> None:
        if self._session != None:
            self._session.close()"""

class Person(base):
    __tablename__ = "people"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    hash = Column(String)
    information = Column(JSON)

class Memory(base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True)
    tag = Column(String)
    content = Column(JSON)

class Secret(base):
    __tablename__ = "secrets"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    encrypted_key = Column(String)