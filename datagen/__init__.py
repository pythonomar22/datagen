"""
DataGen: Synthetic Data Generation for LLM Training
"""

__version__ = "0.1.0"

from datagen.generator import Generator
from datagen.config import Config
from datagen.results import Results

__all__ = ["Generator", "Config", "Results"] 