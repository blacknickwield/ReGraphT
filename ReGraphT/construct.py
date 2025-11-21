import os
import sys
import json
import re
import copy
import argparse
import logging
from typing import Optional

from openai import OpenAI

from ReGraphT.ReGraph import (
    ReGraph, 
    ReGraphNode, 
    ReGraphEdge
)
from ReGraphT.prompt import (
    CUDA_REASONING_SYSTEM_PROMPT,
    CUDA_RELABEL_SYSTEM_PROMPT
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=os.environ["LOG_PATH"],
    filemode="a",
)

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'], 
    base_url=os.environ['BASE_URL']
)

def reason(
    kernel: dict, 
    model: str, 
    temperature: float=0.7,
    max_tokens: int=8192,
    top_p: float=0.9
) -> Optional[list[dict]]:
    """Perform reasoning on a single CUDA kernel using the LLM.
    Returns a trajectory of optimization steps.
    """
    code = {
        "kernel": kernel['kernel']
    }
    logging.info(f"{kernel['index']} Kernel: {kernel['name']}, reasoning start")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CUDA_REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(code)}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    pattern = r'```json\n(.*?)\n```'
    content = response.choices[0].message.content
    matches = re.findall(pattern, content, re.DOTALL)
    if matches is None:
        logging.error(f"Error in {kernel['index']} kernel {kernel['name']}: No matches trajectory found.")
        return None
    
    trajectory = json.loads(matches[0])
    logging.info(f"{kernel['index']} Kernel: {kernel['name']}, reasoning end")
    
    return trajectory


def relabel(
    trajectory: dict, 
    re_graph: ReGraph, 
    model: str, 
    temperature: float=0.7,
    max_tokens: int=8192,
    top_p: float=0.9
) -> dict:
    """
    Relabel the optimization trajectory according to existing methods in ReGraph.
    """
    
    trajectory_ = copy.deepcopy(trajectory)
    methods = []
    for re_graph_node in re_graph.regraph_nodes:
        methods.append(re_graph_node.name)
    data = {
        "methods": methods,
        "process": trajectory
    }
    logging.info(f"trajectory relabel start")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CUDA_RELABEL_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(data)}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    
    pattern = r'```json\n(.*?)\n```'
    content = response.choices[0].message.content
    matches = re.findall(pattern, content, re.DOTALL)
    if matches is None:
        logging.error(f"Error in relabel: No matches relabels found.")
        return None
    
    labels = json.loads(matches[0])
    
    if len(labels) != len(trajectory_):
        logging.error(f"Error in relabel: The length of relabels is not equal to the length of trajectory.")
        return None
    
    # Apply relabels to trajectory
    for idx, label in enumerate(labels):
        if label['existed'] == 'yes':
            if label['method'] not in methods:
                logging.error(f"Error in relabel: The relabel method {label['method']} is not in the existed methods.")
                return None
            trajectory_[idx]['method'] = label['method']
        elif label['existed'] == 'no':
            pass
    logging.info(f"trajectory relabel end")
    return trajectory_


def merge(kernel: dict, trajectory: list[dict], re_graph: ReGraph):
    """
    Merge a kernel's trajectory into the current ReGraph.
    """
    name = kernel['name']
    code = kernel['kernel']
    index = kernel['index']
    
    logging.info(f"{index} kernel: {name} merge start")
    re_graph.merge(name=name, code=code, trajectory=trajectory)
    logging.info(f"{index} kernel: {name} merge end")
    
    
def save_re_graph(re_graph: ReGraph, save_dir: str, prefix: str, steps: int, final: bool=False):
    """
    Save the ReGraph to a JSON file.
    """
    if not final:
        save_path = os.path.join(save_dir, f"{prefix}_{steps}.json")
    else:
        save_path = os.path.join(save_dir, f"{prefix}_final.json")
    logging.info(f"ReGraph save steps: {steps}, save path: {save_path}")
    re_graph.save(save_path=save_path)
    logging.info(f"ReGraph save finished")
    

def construct_regraph(args):
    """
    Construct ReGraph using LLM.
    """
    re_graph_path = args.re_graph
    if re_graph_path is None:
        re_graph = ReGraph()
    else:
        with open(re_graph_path, 'r') as f:
            graph = json.load(f)
        re_graph = ReGraph.from_graph(graph)
        
    # sequence kernels
    kernel_path = args.kernel_path
    with open(kernel_path, 'r') as f:
        kernels = [json.loads(line) for line in f.readlines()]
    
    # LLM parameters 
    model = args.model
    temperature: float = args.temperature
    top_p: float = args.top_p
    top_k: int = args.top_k
    max_tokens: int = args.max_tokens
    
    # ReGraph saving parameters
    save_steps = args.save_steps
    save_dir = args.save_dir
    prefix = args.prefix
    
    steps = 0
    # Process each kernel and update ReGraph
    for kernel in kernels:
        try:
            # 1. Generate optimization trajectory using LLM
            trajectory = reason(
                kernel=kernel, 
                model=model, 
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            if trajectory is None:
                continue
            # 2. Relabel trajectory using existing ReGraph methods
            trajectory_ = relabel(
                trajectory=trajectory, 
                re_graph=re_graph, 
                model=model, 
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            if trajectory_ is None:
                continue
            
            # 3. Merge trajectory into ReGraph
            merge(kernel=kernel, trajectory=trajectory_, re_graph=re_graph)

            # 4. Save ReGraph periodically
            steps += 1
            if save_steps > 0 and steps % save_steps == 0:
                save_re_graph(re_graph=re_graph, save_dir=save_dir, prefix=prefix, steps=steps)
        except Exception as e:
            logging.error(f"Error in {kernel['index']} kernel {kernel['name']}: {e}")
            continue
        
    save_re_graph(re_graph=re_graph, save_dir=save_dir, prefix=prefix, steps=steps, final=True)
    logging.info(f"ReGraph saved to {save_dir} with prefix {prefix} at step {steps}.")


def parser_args():
    parser = argparse.ArgumentParser(description="ReGraph Construction")
    parser.add_argument('--re_graph', type=str, default=None, required=False, help='ReGraph path')
    parser.add_argument('--kernel_path', type=str, default=None, required=True, help='kernel path')
    parser.add_argument('--save_steps', type=int, default=10, required=False, help='save steps')
    parser.add_argument('--save_dir', type=str, required=True, help='ReGraph save dir')
    parser.add_argument('--prefix', type=str, default='ReGraph', required=False, help='ReGraph saved prefix')
    parser.add_argument('--model', type=str, default='deepseek-chat', required=False, help='LLM model')
    parser.add_argument('--temperature', type=float, default=0.7, required=False, help='temperature')
    parser.add_argument('--max_tokens', type=float, default=8192, required=False, help='max_tokens')
    parser.add_argument('--top_p', type=float, default=0.9, required=False, help='top_p')
    parser.add_argument('--top_k', type=int, default=-1, required=False, help='top_k')
    
    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    construct_regraph(args)


if __name__ == "__main__":
    main()
