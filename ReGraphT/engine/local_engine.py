from typing import Union, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .inference_engine import (
    EngineConfig, 
    SamplingParams, 
    InferenceEngine,
    register_engine
)

__all__ = ['LocalEngine']

@register_engine
class LocalEngine(InferenceEngine):
    def __init__(
        self,
        config: EngineConfig
    ):
        super(LocalEngine, self).__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.local_model_path, trust_remote_code=True)
        self.llm = LLM(config.local_model_path)
    
    def generate(
        self, 
        prompts: Union[list[dict], list[list[dict]]],
        config: SamplingParams, 
        **kwargs
    ):
        try:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                log_probs=config.log_probs
            )
            if isinstance(prompts, list) and isinstance(prompts[0], dict):
                prompts = [prompts]
            prompts = [self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
            outputs = self.llm.generate(
                prompts, 
                sampling_params
            )
            
            batch = []
            
            for idx, output in enumerate(outputs):
                prompt = output.prompt
                generation = output.outputs[0].text
                generation_ids = output.outputs[0].token_ids
                logprobs = output.outputs[0].logprobs
                
                prompt_tokens = len(self.llm.get_tokenizer().encode(prompt))
                generation_tokens = len(generation_ids)
                tokens = prompt_tokens + generation_tokens
                item = {
                    'prompt': prompt,
                    'generation': generation,
                    'generation_ids': generation_ids,
                    'logprobs': logprobs,
                    'prompt_tokens': prompt_tokens,
                    'generation_tokens': generation_tokens,
                    'tokens': tokens
                }
                batch.append(item)
                
        except Exception as e:
            print(e)
            raise
            
        return batch