from causal_graph.dynamic_graph import DynCausalGraph


def three_step_stat(temporal_index=0):
    full_vertices = ["X_0", "Z_0", "Y_0", "X_1", "Z_1", "Y_1", "X_2", "Z_2", "Y_2"]
    full_edges = [
        ("X_0", "Z_0"),
        ("Z_0", "Y_0"),
        ("X_1", "Z_1"),
        ("Z_1", "Y_1"),
        ("X_2", "Z_2"),
        ("Z_2", "Y_2"),
        ("X_0", "X_1"),
        ("X_1", "X_2"),
        ("Z_0", "Z_1"),
        ("Z_1", "Z_2"),
        ("Y_0", "Y_1"),
        ("Y_1", "Y_2"),
    ]
    full_treat_vars = [["X_0", "Z_0"], ["X_1", "Z_1"], ["X_2", "Z_2"]]
    full_output_vars = ["Y_0", "Y_1", "Y_2"]
    full_do_vars = [[], [], []]
    dyn_graph = DynCausalGraph(
        full_vertices,
        full_edges,
        full_treat_vars,
        full_output_vars,
        full_do_vars,
        temporal_index,
    )
    return dyn_graph