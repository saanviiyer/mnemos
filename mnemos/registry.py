"""Name -> memory-module registry, so configs can name a memory by string."""
from __future__ import annotations

from typing import Callable, Dict, Type

MEMORY_REGISTRY: Dict[str, Type] = {}


def register_memory(name: str) -> Callable:
    def wrap(cls):
        if name in MEMORY_REGISTRY:
            raise KeyError(f"memory kind {name!r} already registered")
        MEMORY_REGISTRY[name] = cls
        cls.kind = name
        return cls

    return wrap


def build_memory(kind: str, d_model: int, **params):
    if kind not in MEMORY_REGISTRY:
        raise KeyError(f"unknown memory kind {kind!r}; known: {sorted(MEMORY_REGISTRY)}")
    return MEMORY_REGISTRY[kind](d_model=d_model, **params)


def available() -> list[str]:
    return sorted(MEMORY_REGISTRY)
