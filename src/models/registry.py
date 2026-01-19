from typing import Dict, Type, Any
import torch.nn as nn
from omegaconf import DictConfig

class ModelFactory:
    _REGISTRY: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type[nn.Module]):
            cls._REGISTRY[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def get_model(cls, cfg: DictConfig) -> nn.Module:
        """
        Instantiate a model based on the configuration.

        Args:
            cfg: The configuration object, which must contain a 'model' section with a 'name' attribute.

        Returns:
            nn.Module: The instantiated model.

        Raises:
            ValueError: If 'model' section or 'name' attribute is missing, or if the model name is not registered.
        """
        if not hasattr(cfg, 'model'):
            raise ValueError("Config does not have a 'model' section.")
            
        model_cfg = cfg.model
        
        if not hasattr(model_cfg, 'name'):
            raise ValueError("Model config must specify 'name'.")
             
        model_name = model_cfg.name
        
        if model_name not in cls._REGISTRY:
             raise ValueError(f"Model '{model_name}' not registered. Available: {list(cls._REGISTRY.keys())}")
             
        return cls._REGISTRY[model_name](cfg)
