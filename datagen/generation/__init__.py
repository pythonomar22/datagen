"""
Generation module for synthetic data
"""

from datagen.generation.self_instruct import SelfInstructGenerator
from datagen.generation.evol_instruct import EvolInstructGenerator

__all__ = ["SelfInstructGenerator", "EvolInstructGenerator"] 