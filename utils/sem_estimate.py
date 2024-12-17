from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp
from utils.gaussian_process import (
    build_gprm,
    build_gaussian_variable,
    build_gaussian_process,
)
import networkx as nx


def label_pairs(D_obs: OrderedDict, node: str, predecessors: list) -> tuple:
    """Given the observation dataset, the target node, and its predecessors, this function will return the observation data
    for the target node and its predecessors.

    Args:
        D_obs (OrderedDict): observation dataset
        node (str): target node
        predecessors (list): list of predecessors

    Returns:
        tuple: observation data predecessors and target node
    """
    node_name, node_index = node.split("_")
    obs_data = []
    for parent in predecessors:
        parent_name, parent_index = parent.split("_")
        obs_data.append(D_obs[parent_name][:, int(parent_index)])
    obs_data_x = tf.stack(obs_data, axis=1)
    obs_data_y = D_obs[node_name][:, int(node_index)]
    return obs_data_x, obs_data_y


def fcns4sem(the_graph: object, D_obs: OrderedDict, temporal_index: int = None) -> dict:
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
        node_index = int(node.split("_")[1])
        node_name = node.split("_")[0]
        if temporal_index is not None and node_index != temporal_index:
            continue
        predecessors = list(the_graph.predecessors(node))
        if not predecessors:
            obs_data = D_obs[node_name][:, int(node_index)]
            # simply consider these nodes follow a normal distribution
            # with the mean and std calculated from the observation data
            # practitioners can replace this with more sophisticated models
            fcns[node_index][node_name] = build_gaussian_variable(obs_data)
        else:
            obs_data_x, obs_data_y = label_pairs(D_obs, node, predecessors)
            index_ini = tf.ones((1, len(predecessors)))
            # build the Gaussian Process Regression Model
            gprm, _, _ = build_gprm(index_x=index_ini, x=obs_data_x, y=obs_data_y)
            fcns[node_index][node_name] = build_gaussian_process(gprm, predecessors)
    return fcns


def fy_and_fny(
    the_graph: object,
    D_obs: OrderedDict,
    target_node_name: str,
    temporal_index: int = None,
) -> tuple:
    """Given the observation dataset, this function will fit the
    emission and transition functions for the target node at each time step.

    Args:
        the_graph (object): graph object for current time step
        D_obs (OrderedDict): observation dataset
        target_node_name (str): target node name
        temporal_index (int, optional): temporal index. Defaults to None. If None,
        the function will fit the functions for all time steps.

    Returns:
        tuple: emission and transition dictionaries that contains mean and 
        covariance functionsat each time step (if temporal_index is None)
        tuple: emission and transition mean and covariance functions at the 
        specified time step (if temporal_index is not None)
    """
    # sort the nodes by topological order
    sorted_nodes = list(nx.topological_sort(the_graph))
    # find the maximum temporal index
    max_temporal_index = int(sorted_nodes[-1].split("_")[1])
    assert (max_temporal_index + 1) == D_obs[sorted_nodes[0].split("_")[0]].shape[
        1
    ], "Temporal index mismatch"
    fy_fcns = {(t): OrderedDict() for t in range(max_temporal_index + 1)}
    fny_fcns = {(t): OrderedDict() for t in range(max_temporal_index + 1)}

    for t in range(max_temporal_index):
        if temporal_index is not None and t != temporal_index:
            continue
        current_node = target_node_name + "_" + str(t)
        predecessors = list(the_graph.predecessors(current_node))
        y_nodes = []
        ny_nodes = []
        assert predecessors, "The target node has no predecessors."
        for predecessor in predecessors:
            predecessor_index = int(predecessor.split("_")[1])
            if predecessor_index < t:
                y_nodes.append(predecessor)
            elif predecessor_index == t:
                ny_nodes.append(predecessor)
        if y_nodes:
            # if the target node has predecessors at previous time steps
            # then based on assumption 1, the target node is impacted by two functions, i.e., transition and emission functions
            y_nodes_current = [
                node.split("_")[0] + "_" + str(int(node.split("_")[1]) + 1)
                for node in y_nodes
            ]
            obs_data_x, obs_data_y = label_pairs(D_obs, current_node, y_nodes_current)
            index_ini = tf.ones((1, len(y_nodes_current)))
            # build the Gaussian Process Regression Model
            gprm, _, _ = build_gprm(index_x=index_ini, x=obs_data_x, y=obs_data_y)
            mean_fcn = lambda new_index: tf.squeeze(gprm.get_marginal_distribution(new_index).mean())
            std_fcn = lambda new_index: tf.squeeze(gprm.get_marginal_distribution(new_index).stddev())
            fy_fcns[t] = [mean_fcn, std_fcn]

            # build the emission function
            obs_data_x, obs_data_y = label_pairs(D_obs, current_node, ny_nodes)
            obs_data_x0, _ = label_pairs(D_obs, current_node, y_nodes)
            fy_pred = fy_fcns[t][0](obs_data_x0)
            obs_data_fny = obs_data_y - fy_pred

            index_ini = tf.ones((1, len(ny_nodes)))
            # build the Gaussian Process Regression Model
            gprm, _, _ = build_gprm(index_x=index_ini, x=obs_data_x, y=obs_data_fny)
            mean_fcn = lambda new_index: tf.squeeze(gprm.get_marginal_distribution(new_index).mean())
            std_fcn = lambda new_index: tf.squeeze(gprm.get_marginal_distribution(new_index).stddev())
            fny_fcns[t] = [mean_fcn, std_fcn]

        else:
            assert len(ny_nodes) == len(predecessors)
            # if the target node has no predecessors at previous time steps
            # then the target node is impacted by a single emission functions
            obs_data_x, obs_data_y = label_pairs(D_obs, current_node, predecessors)
            index_ini = tf.ones((1, len(predecessors)))
            # build the Gaussian Process Regression Model
            gprm, _, _ = build_gprm(index_x=index_ini, x=obs_data_x, y=obs_data_y)
            mean_fcn = lambda new_index: tf.squeeze(gprm.get_marginal_distribution(new_index).mean())
            std_fcn = lambda new_index: tf.squeeze(gprm.get_marginal_distribution(new_index).stddev())
            fny_fcns[t] = [mean_fcn, std_fcn]

            fy_fcns[t] = [None, None]
            

    if temporal_index is None:
        return fy_fcns, fny_fcns
    else:
        return (
            fy_fcns[temporal_index],
            fny_fcns[temporal_index],
        )


def sem_hat(fcns) -> classmethod:

    class SEMhat:
        def __init__(self):
            self.fcns = fcns
            self.sorted_nodes = list(fcns[0].keys())

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