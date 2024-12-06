import networkx as nx
from causal_graph.utils import is_subset, power_set

class CausalGraph:
    def __init__(self, vertices: list[str], edges: list[tuple[str, str]], treat_vars: list[str], output_var: str):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(vertices)
        self.graph.add_edges_from(edges)
        assert is_subset(treat_vars, vertices), "Treatment variables must be a subset of vertices."
        assert output_var in vertices, "Output variable must be in vertices."
        assert output_var not in treat_vars, "Output variable must not be in treatment variables."
        assert nx.is_directed_acyclic_graph(self.graph), "Graph must be a directed acyclic graph."
        self.treat_vars = treat_vars
        self.output_var = output_var
        self.non_manip_vars = [x for x in vertices if x not in treat_vars+[output_var]]

    def do(self, do_vars: list[str]):
        # do_vars: list of variables to intervene on
        # assert is_subset(do_vars, self.treat_vars), "Intervention variables must be a subset of treatment variables."
        do_graph = self.graph.copy()
        # delete edges pointing to do_vars
        for node in do_vars:
            for predecessor in list(do_graph.predecessors(node)):
                do_graph.remove_edge(predecessor, node) if predecessor is not None else None
        # delete vertices that are not able to reach the output_var
        anc_node = list(nx.ancestors(do_graph, self.output_var))
        for node in list(do_graph.nodes):
            if node not in anc_node+[self.output_var]:
                do_graph.remove_node(node)
        return do_graph
    
    def in_do_graph(self, do_vars: list[str]):
        do_graph = self.do(do_vars)
        return is_subset(do_vars, list(do_graph.nodes))

    def minimal_interven_set(self) -> list[list[str]]:
        # minimal intervention set using a greedy algorithm, check which element of power set is in do_graph
        # algorithm still needs optimization
        close_idx = []
        for treat_var in self.treat_vars:
            try:
                close_idx.append(nx.shortest_path_length(self.graph, treat_var, self.output_var))
            except nx.exception.NetworkXNoPath:
                print(f"No path from {treat_var} to {self.output_var}, removing {treat_var} from treatment variables.")
        sorted_treat_vars = [x for _, x in sorted(zip(close_idx, self.treat_vars), key=lambda pair: pair[0])]

        mis = []
        for subset in power_set(sorted_treat_vars):
            if self.in_do_graph(subset):
                mis.append(subset)
        return mis