"""mnemos: large memory models, with the controls attached."""
from .config import DataConfig, ExperimentConfig, MemoryConfig, ModelConfig, TrainConfig
from .model import MemoryLM
from .modules import KNNMemory, MemoryModule, ProductKeyMemory, SlotMemory, SurpriseMemory
from .registry import available, build_memory, register_memory

__version__ = "0.1.0"
__all__ = [
    "ExperimentConfig", "ModelConfig", "MemoryConfig", "DataConfig", "TrainConfig",
    "MemoryLM", "MemoryModule", "ProductKeyMemory", "SlotMemory", "SurpriseMemory",
    "KNNMemory", "register_memory", "build_memory", "available",
]
