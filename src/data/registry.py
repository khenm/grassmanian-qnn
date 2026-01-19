from pydantic import BaseModel, validator
from omegaconf import DictConfig
from typing import Dict, Type, Any

class HSIDatasetConfig(BaseModel):
    name: str
    data_path: str
    gt_path: str = ""
    patch_size: int = 5
    batch_size: int = 64
    spectral_bands: int
    train_split: float = 0.8
    components: int = 3
    split_mode: str = "easy"
    root_dir: str = ""
    tar_dir: str = ""

    @validator('patch_size')
    def validate_patch(cls, v):
        if v % 2 == 0:
            raise ValueError("Patch size must be odd.")
        return v

class DataFactory:
    _REGISTRY: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type):
            cls._REGISTRY[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def get_datamodule(cls, cfg: DictConfig) -> Any:
        # 1. Convert Hydra config to Pydantic for validation
        dataset_cfg = cfg.dataset if 'dataset' in cfg else cfg
        
        # Convert DictConfig to dict for Pydantic
        cfg_dict = {}
        for k in dataset_cfg:
            try:
                cfg_dict[k] = dataset_cfg[k]
            except Exception:
                pass # skip non-serializable if any
        
        schema = HSIDatasetConfig(**cfg_dict)
        
        # 2. Instantiate the registered class
        if schema.name not in cls._REGISTRY:
            raise ValueError(f"Dataset {schema.name} not registered. Available: {list(cls._REGISTRY.keys())}")
        
        return cls._REGISTRY[schema.name](schema)

    @classmethod
    def get_dataset_class(cls, name: str) -> Type:
        if name not in cls._REGISTRY:
             raise ValueError(f"Dataset {name} not registered. Available: {list(cls._REGISTRY.keys())}")
        return cls._REGISTRY[name]
