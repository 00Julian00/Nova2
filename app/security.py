"""
Description: This script holds various functions to handle security related tasks.
"""

from cryptography.fernet import Fernet
import keyring

from .database_manager import SecretsDatabaseManager

class KeyManager:
    def __init__(self):
        self._secrets_db_manager = SecretsDatabaseManager()
        if not self._get_encryption_key():
            self._set_encryption_key(self._generate_encryption_key())

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
    
    def add_secret(self, name: str, key: str) -> None:
        encrypted_secret = self._encrypt_secret(key)
        self._secrets_db_manager.add_secret(name, encrypted_secret)

    def get_secret(self, name: str) -> str | None:
        encrypted_secret = self._secrets_db_manager.get_secret(name)
        if encrypted_secret:
            return self._decrypt_secret(encrypted_secret)
        else:
            return None
        
    def edit_secret(self, name: str, key: str) -> None:
        encrypted_secret = self._encrypt_secret(key)
        self._secrets_db_manager.edit_secret(name, encrypted_secret)

    def delete_secret(self, name: str) -> None:
        self._secrets_db_manager.delete_secret(name)
