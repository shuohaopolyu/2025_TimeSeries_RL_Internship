import networkx as nx
from graphs.utils import is_subset

class CasualGraph:
    def __init__(self, vertices: list, edges: list, treat_vars: list, output_var: str):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(vertices)
        self.graph.add_edges_from(edges)
        assert is_subset(treat_vars, vertices), "Treatment variables must be a subset of vertices"
        assert output_var in vertices, "Output variable must be in vertices"
        self.treat_vars = treat_vars
        self.output_var = output_var
        self.non_manip_vars = [x for x in vertices if x not in treat_vars+[output_var]]

    def do(self, do_vars: list):
        # do_vars: list of variables to intervene on
        assert is_subset(do_vars, self.treat_vars), "Intervention variables must be a subset of treatment variables"
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
    
    def in_do_graph(self, do_vars: list):
        do_graph = self.do(do_vars)
        return is_subset(do_vars, list(do_graph.nodes))
    
    def power_set(self, lst):
        if len(lst) == 0:
            return [[]]
        return self.power_set(lst[1:]) + [[lst[0]] + x for x in self.power_set(lst[1:])]

    def mis(self):
        # minimal intervention set using a greedy algorithm, check which element of power set is in do_graph
        # algorithm still needs to be optimized.
        close_idx = [nx.shortest_path_length(self.graph, treat_var, self.output_var) for treat_var in self.treat_vars]
        sorted_treat_vars = [x for _, x in sorted(zip(close_idx, self.treat_vars), key=lambda pair: pair[0])]
        for subset in self.power_set(sorted_treat_vars):
            if self.in_do_graph(subset):
                print(f"Minimal intervention set: {subset}")
        return subset