import abc

from ReGraphT.engine import (
    EngineType, 
    EngineConfig, 
    SamplingParams, 
    InferenceEngine
)

__all__ = ['Reasoner']

class Reasoner(abc.ABC):
    def __init__(
        self,
        engine: InferenceEngine,
    ):
        super(Reasoner, self).__init__()
        self.engine = engine
        
    @abc.abstractmethod
    def optimize(
        self,
        kernel: dict,
        *args,
        **kwargs,
    ) -> dict:
        ...
