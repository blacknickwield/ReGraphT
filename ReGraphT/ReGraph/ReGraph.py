import json
from dataclasses import dataclass, field
from typing import Optional

__all__ = ['ReGraphEdge', 'ReGraphNode', 'ReGraph']

@dataclass
class ReGraphEdge(object):
    """ReGraphEdge denotes the state transition between two distinct optimization methods.
    src: ReGraphNode corresponding to the source optimization method
    tgt: ReGraphNode corresponding to the target optimization method
    examples: Specific optimization instances illustrating the transition between the two state
    """
    src: int
    tgt: int
    examples: list[dict] = field(default_factory=list)
    
    def add_example(self, example: dict):
        self.examples.append(example)


@dataclass
class ReGraphNode(object):
    """ReGraphNode denotes an optimization method within ReGraph.
    index: Index of the optimization method
    name: Name of the optimization method
    in_edges: `ReGraphEdge` instances that terminate at this node
    out_edges: `ReGraphEdge` instances that originate from this node
    """
    index: int
    name: str
    in_edges: list[ReGraphEdge] = field(default_factory=list) 
    out_edges: list[ReGraphEdge] = field(default_factory=list) 
    
    def __str__(self):
        return f'{self.index} {self.name}'

    def add_in_edge(self, edge: ReGraphEdge):
        self.in_edges.append(edge)
        
    def add_out_edge(self, edge: ReGraphEdge):
        self.out_edges.append(edge)
        
class ReGraph(object):
    def __init__(
        self,
        regraph_nodes: list[ReGraphNode]=[], 
        regraph_edges: list[ReGraphEdge]=[]
    ):
        """ReGraph is a directed graph describing how optimization methods relate to each
        other and how code can be transformed through a sequence of optimization steps.

        The graph is dynamic: each new optimization trajectory discovered by an LLM can
        be merged into the graph, allowing ReGraph to accumulate reusable optimization
        knowledge over time.
        regraph_nodes: Preconstructed list of nodes.
        regraph_edges: Preconstructed list of edges.
        """
        if len(regraph_nodes) > 0 and len(regraph_edges) > 0:
            self.regraph_nodes = regraph_nodes
            self.regraph_edges = regraph_edges
            
            # Construct the initial node, which serves as the code’s initial state prior to any optimization (the sequential version)
            self.init_state = self.regraph_nodes[0]
        else:
            self.regraph_nodes = []
            self.regraph_edges = []
        
            # Construct the initial node, which serves as the code’s initial state prior to any optimization (the sequential version)
            self.init_state = ReGraphNode(0, "init state")
            self.regraph_nodes.append(self.init_state)
            
        # Current state in ReGraph
        self.state = self.init_state
        
    def reset(self):
        """
        Reset the current traversal state to the initial state.
        """
        self.state = self.init_state
        
    @staticmethod
    def from_graph(graph: dict):
        """
        Construct a ReGraph object from a JSON-compatible dictionary.
        """
        nodes = graph['node']
        edges = graph['edge']
        regraph_nodes = []
        regraph_edges = []
        for edge in edges:
            regraph_edge = ReGraphEdge(src=edge['src'], tgt=edge['tgt'], examples=edge['examples'])
            regraph_edges.append(regraph_edge)
        for node in nodes:
            regraph_node = ReGraphNode(node['index'], node['name'])
            tgt = node['index']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            for src in node['in']:
                for edge in regraph_edges:
                    if edge.src == src and edge.tgt == tgt:
                        regraph_node.in_edges.append(edge)
                        break
            src = node['index']
            for tgt in node['out']:
                for edge in regraph_edges:
                    if edge.src == src and edge.tgt == tgt:
                        regraph_node.out_edges.append(edge)
            regraph_nodes.append(regraph_node)
        regraph = ReGraph(regraph_nodes=regraph_nodes, regraph_edges=regraph_edges)
        return regraph
    
    def __str__(self):
        node_str = f"node:\n{"\n".join((str(node) for node in self.regraph_nodes))}"
        edge_str = f"edge:\n{"\n".join((f'{edge.src}->{edge.tgt}' for edge in self.regraph_edges))}"
        return "\n".join([node_str, edge_str])
    
    def merge(self, name: str, code: str, trajectory: list[dict]):
        """
        Merge an LLM-generated optimization trajectory into the existing ReGraph.
        """
        state: ReGraphNode = self.init_state
        last_code = code
        for step in trajectory:
            # 1. Determine whether the current state already has an outgoing edge 
            # that transitions to the next optimization method
            merged = False
            for edge in state.out_edges:
                tgt: ReGraphNode = self.regraph_nodes[edge.tgt]
                if tgt.name == step['method']:
                    merged = True
                    edge.examples.append({
                        "name": name,
                        "think": step['think'],
                        "detail": step['detail'],
                        "before": last_code,
                        "after": step['code']
                    })
                    state = tgt # State transition
                    last_code = step['code']
                    break
            # 2. If the current state does not have any edges pointing to an optimized state, 
            # verify whether the optimization method is present in the current ReGraph
            if not merged:
                existed = False
                optimization_node: ReGraphNode = None
                for node in self.regraph_nodes:
                    if node.name == step['method']:
                        optimization_node = node
                        existed = True
                        break
                # 2.1 When the optimization method is present in the current ReGraph, 
                # create a new edge
                if existed:
                    edge = ReGraphEdge(src=state.index, tgt=optimization_node.index)
                    edge.add_example({
                        "name": name,
                        "think": step['think'],
                        "detail": step['detail'],
                        "before": last_code,
                        "after": step['code']
                    })
                    self.regraph_edges.append(edge)
                    # TODO: Encapsulate the post-edge-addition operations
                    state.add_out_edge(edge)
                    optimization_node.add_in_edge(edge)
                    state = optimization_node
                    last_code = step['code']
                # 2.2 When the optimization method is not present in the current ReGraph, 
                # create it and add a new edge.
                else:
                    index = len(self.regraph_nodes)
                    optimization_node = ReGraphNode(index=index, name=step['method'])
                    self.regraph_nodes.append(optimization_node)
                    edge = ReGraphEdge(src=state.index, tgt=optimization_node.index)
                    # TODO: Encapsulate the post-edge-addition operations
                    edge.add_example({
                        "name": name,
                        "think": step['think'],
                        "detail": step['detail'],
                        "before": last_code,
                        "after": step['code'],
                    })
                    state.add_out_edge(edge)
                    optimization_node.add_in_edge(edge)
                    state = optimization_node
                    last_code = step['code']

    def save(self, save_path: str):
        """
        Serialize the ReGraph into a JSON file.
        """
        re_graph = {
            "node": [],
            "edge": [],
        }
        
        for node in self.regraph_nodes:
            node_dict = {
                "index": node.index,
                "name": node.name,
                "in": [edge.src for edge in node.in_edges],
                "out": [edge.tgt for edge in node.out_edges],
            }
            re_graph["node"].append(node_dict)
        for edge in self.regraph_edges:
            edge_dict = {
                "src": edge.src,
                "tgt": edge.tgt,
                "examples": edge.examples,
            }
            re_graph["edge"].append(edge_dict)
        
        with open(save_path, 'w') as f:
            json.dump(re_graph, f, indent=4)
        
    def get_state(self, index: int) -> Optional[ReGraphNode]:
        """
        Retrieve a node by its index.
        """
        if index >= len(self.regraph_nodes):
            return None
        return self.regraph_nodes[index]
