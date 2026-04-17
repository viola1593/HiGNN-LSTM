"""TWSA HiGNN-LSTM: Global Water Storage Prediction Models."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

from . import datamodules
from . import models
from . import losses
from . import evaluation
from . import utils

__all__ = [
    "datamodules",
    "models",
    "losses",
    "evaluation",
    "utils",
]
