import networkx as nx
from graphs.causalgraph import CausalGraph


class DynCausalGraph(CausalGraph):
    def __init__(
        self,
        full_vertices: list[str],
        full_edges: list[tuple[str, str]],
        full_treat_vars: list[list[str]],
        full_do_vars: list[list[str]],
        full_output_vars: list[str],
        temporal_index: int,
    ):
        self.full_vertices = full_vertices
        self.full_edges = full_edges
        self.full_treat_vars = full_treat_vars
        self.full_do_vars = full_do_vars
        self.full_output_vars = full_output_vars
        self._temporal_index = None
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
        self.graph = self.do(self.full_do_vars[new_temporal_index])
        
        
