from typing import Union, Optional

import openai

from .inference_engine import (
    EngineConfig, 
    SamplingParams, 
    InferenceEngine,
    register_engine
)

__all__ = ['RemoteEngine']

@register_engine
class RemoteEngine(InferenceEngine):
    def __init__(
        self,
        config: EngineConfig
    ):
        super(RemoteEngine, self).__init__(config)
        self.client = openai.OpenAI(base_url=config.base_url)
        
        
    def generate(
        self,
        prompts: Union[list[dict], list[list[dict]]],        
        config: SamplingParams, 
        **kwargs
    ):
        try:
            if isinstance(prompts, list) and isinstance(prompts[0], dict):
                messages = [prompts]
            else:
                messages = prompts
            
            batch = []
            
            for idx, message in enumerate(messages):
                response = self.client.chat.completions.create(
                    model=config.model,
                    messages=message,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    logprobs=config.log_probs is not None,
                    top_logprobs=config.log_probs
                )
                
                for i, choice in enumerate(response):
                    item = {
                        'prompts': prompts[idx],
                        'generation': choice.message.content,
                        'generation_ids': None,
                        'logprobs': choice.logprobs.dict() if choice.logprobs else None,
                        'prompt_tokens': None,
                        'generation_tokens': None,
                        'tokens': None
                    }
                    batch.append(item)
        except Exception as e:
            print(e)
            raise
        
        return batch