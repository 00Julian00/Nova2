"""
Description: Collection of helper functions
"""

import hashlib

import torch

def hash_embedding(embedding: torch.FloatTensor) -> str:
    """
    Hashes an embedding to a 10 character string.
    """
    
    tensor_bytes = embedding.cpu().numpy().tobytes()
    
    sha_hash = hashlib.sha256(tensor_bytes).hexdigest()
    
    compressed_hash = sha_hash[:10]

    return compressed_hash