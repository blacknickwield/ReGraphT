import abc
from typing import Union, Optional
from dataclasses import dataclass
import enum


__all__ = ['EngineType', 'EngineConfig', 'SamplingParams', 'InferenceEngine', 'register_engine']

ENGINE_REGISTRY = {}

class EngineType(enum.Enum):
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


@dataclass
class EngineConfig:
    base_url: Optional[str] = None
    local_model_path: Optional[str] = None


@dataclass
class SamplingParams:
    model: Optional[str] = None
    temperature: float
    max_tokens: int
    top_p: float = 0.9
    top_k: float = 0.7
    log_probs: Optional[int] = None
    

def register_engine(
    engine_type: EngineType
):
    def decorator(cls):
        ENGINE_REGISTRY[engine_type] = cls
        return cls
    
    return decorator


class InferenceEngine(abc.ABC):
    def __init__(
        self,
        config: EngineConfig
    ):
        super().__init__()
        self.config = config
        
    @staticmethod
    def create_engine(
        engine_type: EngineType, 
        config: EngineConfig
    ):
        cls = ENGINE_REGISTRY[engine_type]
        return cls(config)
        
    @abc.abstractmethod
    def generate(
        self,
        prompts: Union[list[dict], list[list[dict]]],
        config: SamplingParams,
        **kwargs
    ) -> list[dict]:
        ...
        
    @abc.abstractmethod
    def rollout(self, state: dict):
        pass
        
    @abc.abstractmethod
    def generate_available_actions(
        self,
        state: dict
    ) -> list[dict]:
        ...

    def extract_state(
        self, 
        action: dict,
        state: dict
        ) -> dict:
        return {}
    
    