"""
Description: Manages the databases and provides a simple interface
"""

from typing import List, Tuple
import uuid
from pathlib import Path
import warnings
from time import time_ns

import torch
from transformers import AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, Range, SearchParams
from qdrant_client.http import models
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .helpers import hash_embedding

#Database setup
db_people_folder = Path(__file__).parent.parent / "db" / "db_people"
db_people_folder.mkdir(parents=True, exist_ok=True)
db_people_location = db_people_folder / "db_people.db"
db_people_engine = create_engine(f"sqlite:///{db_people_location}", echo=False)

db_memories_folder = Path(__file__).parent.parent / "db" / "db_memories"
db_memories_folder.mkdir(parents=True, exist_ok=True)
db_memories_location = db_memories_folder / "db_memories.db"
db_memories_engine = create_engine(f"sqlite:///{db_memories_location}", echo=False)

db_secrets_folder = Path(__file__).parent.parent / "db" / "db_secrets"
db_secrets_folder.mkdir(parents=True, exist_ok=True)
db_secrets_location = db_secrets_folder / "db_secrets.db"
db_secrets_engine = create_engine(f"sqlite:///{db_secrets_location}", echo=False)

base = declarative_base()

class MemoryEmbeddingDatabaseManager:
    def __init__(self):
        self._prepare_database()
    
    def create_new_entry(self, text: str) -> None:
        embedding = self._compute_embedding(text)

        #Prevent duplicate entries
        if self._is_embedding_in_database(self._torch_tensor_to_float_list(embedding)):
            warnings.warn("Similar or exact embedding already exists in memory embedding database.")
            return

        id = self._qdrant_client.get_collection("memory_embeddings").points_count

        self._qdrant_client.upsert(
            collection_name="memory_embeddings",
            points=[
                PointStruct(
                    id=id,
                    vector=embedding,
                    payload={"text": text}
                )
            ]
        )

    def search_semantic(
            self,
            text: str,
            num_of_results: int = 1,
            search_area: int = 0,
            cosine_threshold: float = 0.5
            ) -> List[List[str]] | None:
        """
        Perform a semantic search in the database.

        Arguments:
            text (str): The text to do a semantic search on.
            num_of_results (int): The amount of results that should be returned. Only returns the maximum amount of results that pass the cosine simmilarity threshold. Defaults to 1.
            search_area (int): The amount of earlier and later entries around each result. If set to 0, only the result itself will be returned. Defaults to 0.

        Returns:
            List of string lists. Each string list is a result with the entries around the result in chronological order. Returns None if no results surpassed the cosine simmilarity threshold.
        """

        query_embedding = self._torch_tensor_to_float_list(self._compute_embedding(text=text))

        search_results = self._qdrant_client.query_points(
            collection_name="memory_embeddings",
            query=query_embedding,
            limit=num_of_results
        )

        #Filter out all results that do not surpass the threshold
        results = [
            result for result in search_results.points
            if result.score >= cosine_threshold
        ]

        if len(results) == 0:
            return None

        #The search is finished. The return structure can be built and returned
        if search_area <= 0:
            return [[result.payload["text"]] for result in results]
        
        #Loop through all results and do area queries
        return_list = []

        for result in results:
            return_list.append(self._query_area(result.id, search_area))

        return return_list

    def _query_area(self, center_id: int, size: int) -> List[str]:
        max_id = self._qdrant_client.get_collection("memory_embeddings").points_count - 1

        limit_down = size
        limit_up = size

        start_id = center_id - size

        #Ensure the start is inside the bounds of the db
        if (center_id - size < 0):
            start_id = 0
            limit_down = 0 #Ensure the area shrinks if the query starts at 0 instead of beeing offset upwards
        elif (start_id + limit_up > max_id):
            start_id = max_id
            limit_up = 0 #Ensure the area shrinks if the query is partially greater then the collection size

        search_results = self._qdrant_client.query_points(
            collection_name="memory_embeddings",
            limit=limit_down + limit_up + 1,
            offset=start_id
        )

        return [result.payload["text"] for result in search_results.points]
    
    def _is_embedding_in_database(self, embedding: list[float], similarity_threshold: float = 0.8) -> bool:
        results = self._qdrant_client.query_points(
            collection_name="memory_embeddings",
            query=embedding,
            limit=1,
            score_threshold=similarity_threshold
        )

        return len(results.points) > 0


    def _compute_embedding(self, text: str) -> torch.FloatTensor:
        """
        Computes an embedding for a given text with shape (1024).
        """
        embedding = self._embedding_model.encode(text, task="text-matching")

        return torch.from_numpy(embedding).squeeze()

    def _prepare_database(self) -> None:
        db_location = Path(__file__).parent.parent / "db" / "db_memory_embeddings"

        self._qdrant_client = QdrantClient(path=db_location)

        if not self._qdrant_client.collection_exists("memory_embeddings"):
            self._qdrant_client.create_collection(collection_name="memory_embeddings", vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

        with warnings.catch_warnings(action="ignore"): #Blocks a deprecation warning
            self._embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to("cuda")

    def _torch_tensor_to_float_list(self, embedding: torch.FloatTensor) -> List[float]:
        return embedding.squeeze().cpu().numpy().tolist()

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
        db_location = Path(__file__).parent.parent / "db" / "db_embeddings"

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

    def delete_secret(self, name: str) -> None:
        secret = self._session.query(Secret).filter_by(name=name).first()
        
        try:
            if secret:
                self._session.delete(secret)
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