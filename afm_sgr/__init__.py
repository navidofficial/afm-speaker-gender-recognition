"""AFM interpretable CNN frontend for speaker gender recognition."""

from .config import Config, get_config
from .model import GenderCNN

__all__ = ["Config", "get_config", "GenderCNN"]
