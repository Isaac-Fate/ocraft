from .app import crnn_app

# Import to register the commands
from .split import *
from .train import *

__all__ = [
    "crnn_app",
]
