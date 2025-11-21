import argparse
import os
import logging
import json

from ReGraphT.ReGraph import ReGraph

from ReGraphT.engine import (
    EngineType,
    EngineConfig,
    SamplingParams,
    InferenceEngine,
)

from ReGraphT.reasoner import (
    Reasoner,
    StandardReasoner,
    CoTReasoner,
    CodeRAGReasoner,
    RethinkMCTSReasoner,
    MCTSRAGReasoner,
    ReGraphTReasoner,
    ReGraphTMCGSReasoner
)

from ReGraphT.executor import (
    Executor,
    load_cuda_eval_dataset,
    load_par_eval_dataset
)

def parse_args():
    parser = argparse.ArgumentParser('ReGraphT')
    ################################################## baselines
    parser.add_argument('--method', type=str, choices=['standard', 'CoT', 'RAG', 'RethinkMCTS', 'MCTS-RAG', 'ReGraphT', 'ReGraphT-MCGS'], required=True)
    parser.add_argument('--engine', type=str, choices=['local', 'remote'], required=True)
    parser.add_argument('--local_model_path', type=str, default=None)
    ################################################## engine
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=8196)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--local_regraph_path', type=str)
    ################################################## dataset
    parser.add_argument('--dataset', type=str, choices=['CUDAEval', 'ParEval'], required=True)
    parser.add_argument('--local_dataset_path', type=str, required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()
    if args.engine == 'local':
        engine_type = EngineType.LOCAL
        local_model_path = args.local_model_path
        engine_config = EngineConfig(local_model_path=local_model_path)
    if args.engine == 'remote':
        engine_type = EngineType.REMOTE
        base_url = args.base_url
        engine_config = EngineConfig(base_url=base_url)

    inference_engine = InferenceEngine.create_engine(
        engine_type=engine_type,
        engine_config=engine_config
    )
    
    method = args.method
    if method == 'standard':
        reasoner = StandardReasoner(engine=inference_engine)
    if method == 'CoT':
        reasoner = CoTReasoner(engine=inference_engine)
    if method == 'RAG':
        reasoner = CodeRAGReasoner(engine=inference_engine)
    if method == 'RethinkMCTS':
        reasoner = RethinkMCTSReasoner(engine=inference_engine)
    if method == 'MCTS-RAG':
        reasoner = MCTSRAGReasoner(engine=inference_engine)
    if method == 'ReGraphT':
        with open(args.local_regraph_path, 'r') as f:
            regraph_json = json.load(f)
        regraph = ReGraph.from_graph(regraph_json)
        reasoner = ReGraphTReasoner(
            engine=inference_engine,
            regraph=regraph
        )
    if method == 'ReGraphT-MCGS':
        with open(args.local_regraph_path, 'r') as f:
            regraph_json = json.load(f)
        regraph = ReGraph.from_graph(regraph_json)
        reasoner = ReGraphTMCGSReasoner(
            engine=inference_engine,
            regraph=regraph
        )
    
    if args.dataset == 'CUDAEval':
        dataset = load_cuda_eval_dataset(args.local_dataset_path)
    if args.dataset == 'ParEval':
        dataset = load_par_eval_dataset(args.local_dataset_path)
        
    meta = {}
        
    executor: Executor = Executor.create_executor(
        dataset=args.dataset, 
        agent=reasoner,
        **meta
    )
    
    executor.run(kernels=dataset)
    
if __name__ == '__main__':
    main()
