"""
Description: A script that holds various helper code.
"""

import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
