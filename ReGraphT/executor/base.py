import abc

from ReGraphT.reasoner import Reasoner

__all__ = ['Executor']

EXECUTOR_REGISTRY = {}

def register_executor(
    dataset: str
):
    def decorator(cls):
        EXECUTOR_REGISTRY[dataset] = cls
        return cls
    
    return decorator

class Executor(abc.ABC):
    def __init__(
        self,
        agent: Reasoner,
        **kwargs
):
        super(Executor, self).__init__()
        self.agent: Reasoner = agent
        
    @staticmethod
    def create_executor(
        dataset: str,
        agent: Reasoner,
        **kwargs
    ):
        cls = EXECUTOR_REGISTRY[dataset]
        return cls(agent, **kwargs)
        
    @abc.abstractmethod
    def run(
        self,
        kernels: list[dict],
    ):
        ...
        
    