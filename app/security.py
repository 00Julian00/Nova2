"""
Description: This script holds various functions to handle security related tasks.
"""

from cryptography.fernet import Fernet
import keyring

import huggingface_hub

from .database_manager import SecretsDatabaseManager

class SecretsManager:
    def __init__(self):
        """
        This class is the interface to store and retrieve sensitive information, called secrets.
        Handles encryption and decryption.
        """
        self._secrets_db_manager = SecretsDatabaseManager()
        if not self._get_encryption_key():
            self._set_encryption_key(self._generate_encryption_key())
    
    def add_secret(self, name: str, key: str) -> None:
        """
        Store a new secret. Secret is encrypted automatically.

        Arguments:
            name (str): The name of the secret.
            key (str): The secret itself.
        """
        encrypted_secret = self._encrypt_secret(key)
        self._secrets_db_manager.add_secret(name, encrypted_secret)

    def get_secret(self, name: str) -> str | None:
        """
        Retrive a secret from the database. Will be decrypted automatically.

        Arguments:
            name (str): The name of the secret that should be retrieved.

        Returns:
            str | None: The decrypted value of the secret or None if the secret could not be found.
        """
        encrypted_secret = self._secrets_db_manager.get_secret(name)
        if encrypted_secret:
            return self._decrypt_secret(encrypted_secret)
        
    def edit_secret(self, name: str, key: str) -> None:
        """
        Edit the value of an existing secret.

        Arguments:
            name (str): The name of the secret that should be changed.
            key (str): The new value of the secret. Will be encrypted automatically.
        """
        encrypted_secret = self._encrypt_secret(key)
        self._secrets_db_manager.edit_secret(name, encrypted_secret)

    def delete_secret(self, name: str) -> None:
        """
        Deletes a secret from the database.

        Arguments:
            name (str): The secret that should be deleted.
        """
        self._secrets_db_manager.delete_secret(name)

    def huggingface_login(self, overwrite: bool = False, value: str = "") -> None:
        """
        Attempt to log into huggingface which is required to access restricted repos.
        If a token is already stored, it will be used to log in automatically, if not the user will be prompted in the console to enter their token.
        """
        token = self.get_secret(name="huggingface_token")
        if not token or overwrite:
            token = input("Please enter your huggingface token: ")
            self.add_secret(name="huggingface_token", key=token)
        
        if value != "":
            token = value

        try:
            huggingface_hub.login(token=token)
        except:
            raise Exception("Failed to log into huggingface. Check wether your token is correct and valid and try again.")
        
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