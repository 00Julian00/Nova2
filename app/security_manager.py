"""
Description: This script holds various functions to handle security related tasks.
"""

from cryptography.fernet import Fernet
import keyring

import huggingface_hub

from .database_manager import SecretsDatabaseManager
from .security_data import Secrets

class SecretsManager:
    def __init__(self):
        """
        This class is the interface to store and retrieve sensitive information, called secrets.
        Handles encryption and decryption.
        """
        self._secrets_db_manager = SecretsDatabaseManager()
        if not self._get_encryption_key():
            self._set_encryption_key(self._generate_encryption_key())
    
    def add_secret(self, name: Secrets, key: str) -> None:
        """
        Store a new secret. Secret is encrypted automatically.

        Arguments:
            name (Secrets): The name of the secret.
            key (str): The secret itself.
        """
        encrypted_secret = self._encrypt_secret(key)
        self._secrets_db_manager.add_secret(name.value, encrypted_secret)

    def get_secret(self, name: Secrets) -> str | None:
        """
        Retrive a secret from the database. Will be decrypted automatically.

        Arguments:
            name (Secrets): The name of the secret that should be retrieved.

        Returns:
            str | None: The decrypted value of the secret or None if the secret could not be found.
        """
        encrypted_secret = self._secrets_db_manager.get_secret(name.value)
        if encrypted_secret:
            return self._decrypt_secret(encrypted_secret)
        
    def edit_secret(self, name: Secrets, key: str) -> None:
        """
        Edit the value of an existing secret.

        Arguments:
            name (Secrets): The name of the secret that should be changed.
            key (str): The new value of the secret. Will be encrypted automatically.
        """
        encrypted_secret = self._encrypt_secret(key)

        if not self.get_secret(name=name):
            self._secrets_db_manager.add_secret(name.value, encrypted_secret)
        else:
            self._secrets_db_manager.edit_secret(name.value, encrypted_secret)

    def delete_secret(self, name: Secrets) -> None:
        """
        Deletes a secret from the database.

        Arguments:
            name (Secrets): The secret that should be deleted.
        """
        self._secrets_db_manager.delete_secret(name.value)

    def huggingface_login(self, overwrite: bool = False, token: str = "") -> None:
        """
        Attempt to log into huggingface which is required to access restricted repos.
        Raises an exception if the login fails.
        
        Arguments:
            overwrite (bool): If true, "token" will overwrite the value stored in the database. If false, the database will remain unchanged and "token" will be used to attempt a login, if provided.
            token (str): If provided, this token will be used to log in.
        """
        db_token = self.get_secret(name=Secrets.HUGGINGFACE)

        # Overwrite the value in the db if overwrite.
        if overwrite:
            if db_token:
                self.edit_secret(name=Secrets.HUGGINGFACE, key=token)
            else:
                self.add_secret(name=Secrets.HUGGINGFACE, key=token)

        if token != "":
            try:
                huggingface_hub.login(token=token)
            except Exception as e:
                raise Exception(f"Failed to log into huggingface: {e}")
        elif db_token:
            try:
                huggingface_hub.login(token=db_token)
            except Exception as e:
                raise Exception(f"Failed to log into huggingface: {e}")
        else:
            raise Exception("Failed to log into huggingface. No overwrite value provided and no value found in database.")
        
    def _generate_encryption_key(self) -> bytes:
        return Fernet.generate_key()
    
    def _set_encryption_key(self, key: bytes) -> None:
        keyring.set_password("Nova", "encryption_key", key.decode())
    
    def _get_encryption_key(self) -> bytes | None:
        key_str = keyring.get_password("Nova", "encryption_key")
        if key_str:
            return key_str.encode()
        return None
    
    def _encrypt_secret(self, key: str) -> str:
        fernet = Fernet(self._get_encryption_key())
        return fernet.encrypt(key.encode()).decode()

    def _decrypt_secret(self, encrypted_secret: str) -> str:
        fernet = Fernet(self._get_encryption_key())
        return fernet.decrypt(encrypted_secret.encode()).decode()