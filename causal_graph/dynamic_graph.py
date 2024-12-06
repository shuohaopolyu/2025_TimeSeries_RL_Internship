import networkx as nx
from causal_graph.base import CausalGraph
from causal_graph.utils import check_consistency, check_element_index
import itertools


class DynCausalGraph(CausalGraph):
    def __init__(
        self,
        full_vertices: list[str],
        full_edges: list[tuple[str, str]],
        full_treat_vars: list[list[str]],
        full_output_vars: list[str],
        full_do_vars: list[list[str]],
        temporal_index: int,
    ):
        self.full_vertices = full_vertices
        self.full_edges = full_edges
        self.full_treat_vars = full_treat_vars
        assert check_consistency(full_output_vars), "Output variables are inconsistent."
        assert check_element_index(full_do_vars), "Do variables are inconsistent with time steps."
        self.full_output_vars = full_output_vars
        self._full_do_vars = 0
        self._temporal_index = 0
        self.full_do_vars = full_do_vars
        self.temporal_index = temporal_index

    @property
    def temporal_index(self):
        return self._temporal_index
    
    @temporal_index.setter
    def temporal_index(self, new_temporal_index):
        # Validate the new_temporal_index if necessary
        if not (0 <= new_temporal_index < len(self.full_output_vars)):
            raise ValueError("Time step mismatch the length of output variables.")
        # Update the private variable
        self._temporal_index = new_temporal_index
        # Re-initialize the superclass with updated parameters
        super().__init__(
            self.full_vertices,
            self.full_edges,
            self.full_treat_vars[new_temporal_index],
            self.full_output_vars[new_temporal_index],
        )
        # update the graph according to the new time step
        self.graph = self.do(list(itertools.chain(*self.full_do_vars[:new_temporal_index])))

    @property
    def full_do_vars(self):
        return self._full_do_vars
    
    @full_do_vars.setter
    def full_do_vars(self, new_full_do_vars):
        # Validate the new_full_do_vars if necessary
        if len(new_full_do_vars) != len(self.full_output_vars):
            raise ValueError("Length mismatch between full_do_vars and full_output_vars.")
        # Update the private variable
        self._full_do_vars = new_full_do_vars
        # Re-initialize the superclass with updated parameters
        super().__init__(
            self.full_vertices,
            self.full_edges,
            self.full_treat_vars[self.temporal_index],
            self.full_output_vars[self.temporal_index],
        )
        # update the graph according to the new time step
        self.graph = self.do(list(itertools.chain(*self.full_do_vars[:self.temporal_index])))