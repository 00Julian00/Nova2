"""
Description: Holds all data needed by the security manager.
"""

from enum import Enum

class Secrets(Enum):
    GROQ_API = "groq_api_key"
    ELEVENLABS_API = "elevenlabs_api_key"
    HUGGINGFACE = "huggingface_token"