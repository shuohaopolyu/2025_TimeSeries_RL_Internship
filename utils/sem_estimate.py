from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp
from utils.gaussian_process import (
    build_gprm,
    build_gaussian_variable,
    build_gaussian_process,
)
import networkx as nx


def fcns4sem(the_graph, D_obs: OrderedDict) -> dict:
    """Each node in the dynamic graph can either be generated based on emission or transition functions.
    Given the observation dataset, this function will fit the emission and transition functions for each node.

    Args:
        the_graph (object): graph object for current time step
        D_obs (OrderedDict): observation dataset
        emissions (bool): if True, the function will return the emission functions

    Returns:
        fcns (dict): dictionary of functions at each time step for each node
    """
    # sort the nodes by topological order
    sorted_nodes = list(nx.topological_sort(the_graph))
    # find the maximum temporal index
    max_temporal_index = int(sorted_nodes[-1].split("_")[1])
    assert (max_temporal_index + 1) == D_obs[sorted_nodes[0].split("_")[0]].shape[
        1
    ], "Temporal index mismatch"
    fcns = {t: OrderedDict() for t in range(max_temporal_index + 1)}

    for node in sorted_nodes:
        temproal_index = int(node.split("_")[1])
        node_name = node.split("_")[0]
        predecessors = list(the_graph.predecessors(node))
        if not predecessors:
            obs_data = D_obs[node_name][:, int(temproal_index)]
            # simply consider these nodes follow a normal distribution
            # with the mean and std calculated from the observation data
            # practitioners can replace this with more sophisticated models
            fcns[temproal_index][node_name] = build_gaussian_variable(obs_data)
        else:
            obs_data = []
            for parent in predecessors:
                parent_name, parent_index = parent.split("_")
                obs_data.append(D_obs[parent_name][:, int(parent_index)])
            obs_data_x = tf.stack(obs_data, axis=1)
            obs_data_y = D_obs[node_name][:, int(temproal_index)]
            index_ini = tf.ones((1, len(predecessors)))
            # build the Gaussian Process Regression Model
            gprm, _, _ = build_gprm(index_x=index_ini, x=obs_data_x, y=obs_data_y)
            fcns[temproal_index][node_name] = build_gaussian_process(gprm, predecessors)
    return fcns


def sem_hat(the_graph: object, D_obs: OrderedDict) -> classmethod:

    class SEMhat:
        def __init__(self):
            self.fcns = fcns4sem(the_graph, D_obs)
            self.sorted_nodes = list(self.fcns[0].keys())

        def static(self):
            f = OrderedDict()
            for node in self.sorted_nodes:
                f[node] = (lambda node: lambda sample: self.fcns[0][node](sample))(node)
            return f

        def dynamic(self):
            f = OrderedDict()
            for node in self.sorted_nodes:
                f[node] = (lambda node: lambda t, sample: self.fcns[t][node](sample))(
                    node
                )
            return f

    return SEMhat
