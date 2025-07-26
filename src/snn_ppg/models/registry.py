"""
Model registry mapping model names to their constructors.
Provides a unified factory function for instantiating models by name.
"""
from typing import Any, Type

from torch import nn

from .block import Block
from .dablock import SpikingDABlock
from .spiking_resnet import SpikingResNet

# Registry of available models
MODEL_REGISTRY: dict[str, Type[nn.Module]] = {
    "block": Block,
    "dablock": SpikingDABlock,
    "spiking_resnet": SpikingResNet,
}


def get_model(name: str, *args: Any, **kwargs: Any) -> nn.Module:
    """
    Factory function to retrieve and instantiate a model by name.

    Args:
        name (str): Key name of the model in the registry.
        *args: Positional arguments to forward to the model constructor.
        **kwargs: Keyword arguments for the model constructor.

    Returns:
        nn.Module: An instance of the requested model.

    Raises:
        KeyError: If the model name is not found in the registry.
    """
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        ) from exc
    return cls(*args, **kwargs)
