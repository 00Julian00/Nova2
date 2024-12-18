"""
Description: This script controls the execution of Nova
"""

#! This code is currently only for debbuging purposes.

import torch

class Nova:
    def __init__(self):
        pass

    def start(self):
        #Check for CUDA availability
        if (not torch.cuda.is_available):
            raise Exception("CUDA is required to run Nova.")