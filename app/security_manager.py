"""
Description: This script holds various functions to handle security related tasks.
"""
import dotenv
import os

import huggingface_hub

from Nova2.app.helpers import Singleton

class SecretsManager(Singleton):
    def __init__(self):
        """
        This class is the interface to store and retrieve sensitive information, called secrets.
        Handles encryption and decryption.
        """
        dotenv.load_dotenv("../.env")

    def huggingface_login(self) -> None:
        """
        Attempt to log into huggingface which is required to access restricted repos.
        Raises an exception if the login fails.
        Uses the credentials stored in the .env file.
        """
        token = os.getenv("HUGGINGFACE_TOKEN")
        huggingface_hub.login(token=token)