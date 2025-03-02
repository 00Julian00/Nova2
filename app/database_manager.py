"""
Description: Manages the databases and provides a simple interface
"""

from typing import List, Tuple
import uuid
from pathlib import Path
import warnings
import re

import torch
from transformers import AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client.http import models
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .helpers import suppress_output

# Database setup
db_secrets_folder = Path(__file__).parent.parent / "db" / "db_secrets"
db_secrets_folder.mkdir(parents=True, exist_ok=True)
db_secrets_location = db_secrets_folder / "db_secrets.db"
db_secrets_engine = create_engine(f"sqlite:///{db_secrets_location}", echo=False)

base = declarative_base()

def _is_connection_open(func: callable):
    """
    Prevent access to the database on a closed connection, because concurrent access is not allowed.
    """
    def wrapper(self, *args, **kwargs):
        if not self._qdrant_client:
            raise Exception("The connection must be opened first before interacting with the database.")
        return func(self, *args, **kwargs)
    return wrapper

# RAG hat Swag. LG an Valentin Weyer.
class MemoryEmbeddingDatabaseManager:
    def __init__(self):
        """
        This class is responsible for managing the memory database which stores memories as text-embeddings.
        It also provides a semantic search system used for retrieval augumented generation.
        """
        self._qdrant_client = None
        self._embedding_model = None

    @_is_connection_open
    def create_new_entry(self, text: str) -> None:
        """
        Write new entry to the database. The input is chunked into sentences and each sentence is converted into
        a text embedding.

        Arguments:
            text (str): The text that should be stored to the database.
        """
        split_text = re.split('[.!?]', text) # Split into sentences before storing
        
        for text in split_text:
            self._save_embedding_to_db(text)

    @_is_connection_open
    def _save_embedding_to_db(self, text: str) -> None:
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

    @_is_connection_open
    def search_semantic(
            self,
            text: str,
            num_of_results: int = 1,
            search_area: int = 0,
            cosine_threshold: float = 0.6
            ) -> List[List[str]] | None:
        """
        Perform a semantic search in the database.

        Arguments:
            text (str): The text to do a semantic search on.
            num_of_results (int): The amount of results that should be returned. Only returns the maximum amount of results that pass the cosine simmilarity threshold. Defaults to 1.
            search_area (int): The amount of earlier and later entries around each result. If set to 0, only the result itself will be returned. Defaults to 0.

        Returns:
            list[list[str]]. Each string list is a result with the entries around the result in chronological order. Returns None if no results surpassed the cosine simmilarity threshold.
        """

        query_embedding = self._torch_tensor_to_float_list(self._compute_embedding(text=text))

        search_results = self._qdrant_client.query_points(
            collection_name="memory_embeddings",
            query=query_embedding,
            limit=num_of_results
        )

        # Filter out all results that do not surpass the threshold
        results = [
            result for result in search_results.points
            if result.score >= cosine_threshold
        ]

        if len(results) == 0:
            return None

        # The search is finished. The return structure can be built and returned
        if search_area <= 0:
            return [[result.payload["text"]] for result in results]
        
        # Loop through all results and do area queries
        return_list = []

        for result in results:
            return_list.append(self._query_area(result.id, search_area))

        return return_list

    def _query_area(self, center_id: int, size: int) -> list[str]:
        """
        Query entries around the specified entry to provide more context to the search result of the semantic search.

        Arguments:
            center_id (int): The index of the semantic search result.
            size (int): How many earlier and later entries should be queried. The amount of returned entries is 2 * size + 1.

        Returns:
            list[str]: A list of results from the database.
        """
        max_id = self._qdrant_client.get_collection("memory_embeddings").points_count - 1

        limit_down = size
        limit_up = size

        start_id = center_id - size

        # Ensure the start is inside the bounds of the db
        if (center_id - size < 0):
            start_id = 0
            limit_down = 0 # Ensure the area shrinks if the query starts at 0 instead of beeing offset upwards
        elif (start_id + limit_up > max_id):
            start_id = max_id
            limit_up = 0 # Ensure the area shrinks if the query is partially greater then the collection size

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

        Arguments:
            text (str): The text that will be converted into an embedding.

        Returns:
            torch.FloatTensor: The computed embedding.
        """
        embedding = self._embedding_model.encode(text, task="text-matching")

        return torch.from_numpy(embedding).squeeze()

    def _prepare_database(self) -> None:
        db_location = Path(__file__).parent.parent / "db" / "db_memory_embeddings"

        self._qdrant_client = QdrantClient(path=db_location)

        if not self._qdrant_client.collection_exists("memory_embeddings"):
            self._qdrant_client.create_collection(collection_name="memory_embeddings", vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

        if not self._embedding_model:
            with warnings.catch_warnings(action="ignore"): # Blocks a deprecation warning
                with suppress_output(): # Don't show model downloads
                    self._embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to("cuda")

    def open(self):
        """
        Open the connection to the database. The database can not be accessed before a connection has been opened.
        """
        if self._qdrant_client:
            raise Exception("A previous connection was not closed. Concurrent access is not allowed.")

        try:
            self._prepare_database()
        except:
            raise Exception("Database could not be accessed. You need to close a previous connection before openening a new one. Concurrent access is not allowed.")

    @_is_connection_open
    def close(self):
        """
        Close the connection to the database. Concurrent access is not allowed, so a connection must be closed before a new one can be opened.
        """
        self._qdrant_client = None

    def _torch_tensor_to_float_list(self, embedding: torch.FloatTensor) -> List[float]:
        return embedding.squeeze().cpu().numpy().tolist()


class VoiceDatabaseManager:
    def __init__(self) -> None:
        """
        This class is responsible for managing the database that stores the voice embeddings generated by 'transcriptor.py'.
        It also provides a method to compare two embeddings to determine wether two voices match.
        """
        self._prepare_database()
        self._qdrant_client = None
    
    @_is_connection_open
    def create_voice(self, embedding: torch.FloatTensor, name: str) -> None:
        """
        Creates a new entry in the database.

        Arguments:
            embedding (torch.FloatTensor): The embedding that will be stored in the database.
            name (str): The name of the person the voice belongs to. Will be stored together with the embedding.
        """
        self._qdrant_client.upsert(
            collection_name="voice_embeddings",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=self._torch_tensor_to_float_list(embedding),
                    payload={"name": name}
                )
            ]
        )

    @_is_connection_open
    def create_unknown_voice(self, embedding: torch.FloatTensor) -> str:
        """
        Creates a new voice with the name "UnknownVoiceX", where X is a number starting from 0. These can later be replaced with the correct name, after the system has obtained the name.

        Arguments:
            embedding (torch.FloatTensor): The embedding that will be stored in the database.
        """
        unknown_counter = 0
        while self.does_voice_exist(f"UnknownVoice{unknown_counter}"): # Find an index that is not already in use
            unknown_counter += 1

        self.create_voice(embedding, f"UnknownVoice{unknown_counter}")

        return f"UnknownVoice{unknown_counter}"

    @_is_connection_open
    def get_voice_name_from_embedding(self, embedding: torch.FloatTensor) -> Tuple[str, float] | None:
        """
        Searches for the closest voice embedding to the given embedding.
        Returns the name of the closest voice embedding together with the confidence score.

        Arguments:
            embedding (torch.FloatTensor): The embedding to search for.

        Returns:
            Tuple[str, float] | None: Either returns a tuple with the name of the voice and the confidence score or None if no voice could be found.
        """
        search_results = self._qdrant_client.search(
            collection_name="voice_embeddings",
            query_vector=self._torch_tensor_to_float_list(embedding),
            limit=1
        )

        if len(search_results) > 0:
            return search_results[0].payload["name"], search_results[0].score
        else:
            return None
    
    @_is_connection_open
    def does_voice_exist(self, name: str) -> bool:
        """
        Checks if a voice embedding with the given name exists in the database.

        Arguments:
            name (str): The name to search for.

        Returns:
            bool: Wether a voice with that name already exists in the database.
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

    @_is_connection_open
    def get_voice_id(self, embedding: torch.FloatTensor) -> int | None:
        """
        Searches for the ID of a voice embedding in the database.

        Arguments:
            embedding (torch.FloatTensor): The voice embedding to search for.

        Returns:
            int | None: The index of the voice or None if no voice was found.
        """
        search_results = self._qdrant_client.search(
            collection_name="voice_embeddings",
            query_vector=self._torch_tensor_to_float_list(embedding),
            limit=1
        )

        if len(search_results) > 0:
            return search_results[0].id
        else:
            return None

    @_is_connection_open
    def edit_voice_name(self, old_name: str, new_name: str) -> bool:
        """
        Edit the name of a voice in the database.
        
        Arguments:
            old_name (str): The name to search for.
            new_name (str): What to name the voice.

        Returns:
            bool: Wether the operation was successfull.
        """
        # Find the voice ID using the old name
        search_result = self._qdrant_client.scroll(
            collection_name="voice_embeddings",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="name",
                        match=models.MatchValue(value=old_name)
                    )
                ]
            ),
            limit=1 
        )
        
        if not search_result[0]: # Return False if no voice was found.
            return False
            
        voice_id = search_result[0][0].id
        
        # Update the name
        self._qdrant_client.set_payload(
            collection_name="voice_embeddings",
            payload={"name": new_name},
            points=[voice_id]
        )
        
        return True

    def _prepare_database(self) -> None:
        db_location = Path(__file__).parent.parent / "db" / "db_voice_embeddings"

        self._qdrant_client = QdrantClient(path=db_location)

        if not self._qdrant_client.collection_exists("voice_embeddings"):
            self._qdrant_client.create_collection(collection_name="voice_embeddings", vectors_config=VectorParams(size=512, distance=Distance.COSINE))

    def open(self):
        """
        Open the connection to the database. The database can not be accessed before a connection has been opened.
        """
        if self._qdrant_client:
            raise Exception("A previous connection was not closed. Concurrent access is not allowed.")

        try:
            self._prepare_database()
        except:
            raise Exception("Database could not be accessed. You need to close a previous connection before openening a new one. Concurrent access is not allowed.")

    @_is_connection_open
    def close(self):
        """
        Close the connection to the database. Concurrent access is not allowed, so a connection must be closed before a new one can be opened.
        """
        self._qdrant_client = None

    def _torch_tensor_to_float_list(self, embedding: torch.FloatTensor) -> List[float]:
        return embedding.squeeze().cpu().numpy().tolist()

class SecretsDatabaseManager:
    def __init__(self):
        """
        This class is used to store sensitive information like API keys.
        """
        self._prepare_database()

    def add_secret(self, name: str, encrypted_key: str) -> None:
        """
        Add a new entry to the database.

        Arguments:
            name (str): The name of the secret.
            encrypted_key(str): The secret itself. Will be stored in plain text in the database, so should be encrypted beforehand.
        """
        try:
            new_secret = Secret(name=name, encrypted_key=encrypted_key)
            self._session.add(new_secret)
            self._session.commit()
        except:
            self._session.rollback()
            raise Exception("Error when writing to database.")
        
    def get_secret(self, name: str) -> str | None:
        """
        Retrieve a secret from the database.

        Arguments:
            name (str): The name of the secret that should be retrieved.

        Returns:
            str | None: The secret or None if the secret could not be found.
        """
        secret = self._session.query(Secret).filter_by(name=name).first()

        if secret:
            return secret.encrypted_key
        
    def edit_secret(self, name: str, encrypted_key: str) -> None:
        """
        Modifies the value of a secret.

        Arguments:
            name (str): The name of the secret.
            encrypted_key(str): The secret itself. Will be stored in plain text in the database, so should be encrypted beforehand.
        """
        secret = self._session.query(Secret).filter_by(name=name).first()

        try:
            if secret:
                secret.encrypted_key = encrypted_key
                self._session.commit()
        except:
            self._session.rollback()
            raise Exception("Error when writing to database.")

    def delete_secret(self, name: str) -> None:
        """
        Deletes a secret.

        Arguments:
            name (str): The name of the secret.
        """
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

class Secret(base):
    __tablename__ = "secrets"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    encrypted_key = Column(String)