from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp
from utils.gaussian_process import (
    build_gprm,
    build_gaussian_variable,
    build_gaussian_process,
)
import networkx as nx
import matplotlib.pyplot as plt


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
    node_name, temporal_index = node.split("_")
    obs_data = []
    for parent in predecessors:
        parent_name, parent_index = parent.split("_")
        obs_data.append(D_obs[parent_name][:, int(parent_index)])
    obs_data_x = tf.stack(obs_data, axis=1)
    obs_data_y = D_obs[node_name][:, int(temporal_index)]
    return obs_data_x, obs_data_y

def plot_gprm_results(gprm, obs_data_x, obs_data_y, const=0.1):
    for i in range(obs_data_x.shape[1]):
        min_obs = tf.reduce_min(obs_data_x[:, i])
        max_obs = tf.reduce_max(obs_data_x[:, i])
        min_obs = min_obs - const * (max_obs - min_obs)
        max_obs = max_obs + const * (max_obs - min_obs)
        if i == 0:
            index_new = tf.reshape(tf.linspace(min_obs, max_obs, 1000), (-1, 1))
        else:
            index_new = tf.concat(
                [index_new, tf.reshape(tf.linspace(min_obs, max_obs, 1000), (-1, 1))], axis=1
            )

    mean = gprm.get_marginal_distribution(index_new).mean()
    std = gprm.get_marginal_distribution(index_new).stddev()
    plt.figure()
    plt.plot(obs_data_x, obs_data_y, "ro", label="Observation Data")
    plt.plot(index_new, mean, "b-", label="Mean")
    plt.fill_between(
        index_new[:, 0],
        mean - 2 * std,
        mean + 2 * std,
        color="b",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.legend()
    plt.show()

def fcns4sem(
    the_graph: object, D_obs: OrderedDict, temporal_index: int = None, **gprm_kwargs
) -> dict:
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
    if D_obs[sorted_nodes[0].split("_")[0]].shape[1] < (max_temporal_index + 1):
        print(
            "Warning: The observation data is not enough for the maximum temporal step."
        )

    fcns = {t: OrderedDict() for t in range(max_temporal_index + 1)}

    obs_noise_factor = gprm_kwargs.get("observation_noise_factor", 1e-1)
    learning_rate = gprm_kwargs.get("learning_rate", 2e-4)
    max_training_step = gprm_kwargs.get("max_training_step", 20000)
    debug_mode = gprm_kwargs.get("debug_mode", True)
    intervention_domain = gprm_kwargs.get("intervention_domain", None)

    for node in sorted_nodes:
        temporal_index = int(node.split("_")[1])
        node_name = node.split("_")[0]
        if temporal_index is not None and temporal_index != temporal_index:
            continue
        predecessors = list(the_graph.predecessors(node))
        if not predecessors:
            obs_data = D_obs[node_name][:, temporal_index]
            # simply consider these nodes follow a normal distribution
            # with the mean and std calculated from the observation data
            # practitioners can replace this with more sophisticated models
            fcns[temporal_index][node_name] = build_gaussian_variable(obs_data)
        else:
            obs_data_x, obs_data_y = label_pairs(D_obs, node, predecessors)
            if intervention_domain is not None:
                obs_data_x, obs_data_y, _ = filter_obs_data(obs_data_x, obs_data_y, intervention_domain, predecessors)
            index_ini = tf.ones((1, len(predecessors)))
            # build the Gaussian Process Regression Model
            gprm, _, _ = build_gprm(
                index_x=index_ini,
                x=obs_data_x,
                y=obs_data_y,
                obs_noise_factor=obs_noise_factor,
                learning_rate=learning_rate,
                max_training_step=max_training_step,
                debug_mode=debug_mode,
            )
            if debug_mode:
                plot_gprm_results(gprm, obs_data_x, obs_data_y)
            fcns[temporal_index][node_name] = build_gaussian_process(gprm, predecessors)
    return fcns


def filter_obs_data(obs_data_x: tf.Tensor, obs_data_y: tf.Tensor, intervention_domain: OrderedDict, predecessors: list) -> tuple:
    # given each row of obs_data_x, we can filter out the rows that are outside the intervention domain
    # and the corresponding obs_data_y
    assert len(predecessors) == obs_data_x.shape[1], (len(predecessors), obs_data_x.shape[1])
    mask = tf.ones(obs_data_x.shape[0], dtype=bool)
    for i, parent in enumerate(predecessors):
        parent_name= parent.split("_")[0]
        if parent_name in intervention_domain.keys():
            parent_domain = intervention_domain[parent_name]
            mask = tf.logical_and(mask, tf.logical_and(obs_data_x[:, i] >= parent_domain[0], obs_data_x[:, i] <= parent_domain[1]))
    return obs_data_x[mask], obs_data_y[mask], mask


def fy_and_fny(
    the_graph: object,
    D_obs: OrderedDict,
    target_node_name: str,
    temporal_index: int = None,
    **gprm_kwargs
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
    if D_obs[sorted_nodes[0].split("_")[0]].shape[1] < (max_temporal_index + 1):
        print(
            "Warning: The observation data is not enough for the maximum temporal step."
        )

    fy_fcns = {(t): OrderedDict() for t in range(max_temporal_index + 1)}
    fny_fcns = {(t): OrderedDict() for t in range(max_temporal_index + 1)}

    obs_noise_factor = gprm_kwargs.get("observation_noise_factor", 1e-1)
    learning_rate = gprm_kwargs.get("learning_rate", 2e-4)
    max_training_step = gprm_kwargs.get("max_training_step", 20000)
    debug_mode = gprm_kwargs.get("debug_mode", True)
    intervention_domain = gprm_kwargs.get("intervention_domain", None)

    for t in range(max_temporal_index + 1):
        if temporal_index is not None and t != temporal_index:
            continue
        current_node = target_node_name + "_" + str(t)
        predecessors = list(the_graph.predecessors(current_node))
        y_nodes = []
        ny_nodes = []
        assert predecessors, "The target node has no predecessors."
        for predecessor in predecessors:
            if predecessor.split("_")[0] == target_node_name:
                y_nodes.append(predecessor)
            else:
                ny_nodes.append(predecessor)
        if y_nodes:
            # if the target node has predecessors at previous time steps
            # then based on assumption 1, the target node is impacted by two functions, i.e., transition and emission functions

            def mean_fcn_fy(new_index):
                return tf.reduce_mean(new_index, axis=1)
            
            def std_fcn_fy(new_index):
                return tf.math.reduce_mean(new_index, axis=1) * 1e-4

            fy_fcns[t] = [mean_fcn_fy, std_fcn_fy]

            # build the emission function
            obs_data_x, obs_data_y = label_pairs(D_obs, current_node, ny_nodes)
            if intervention_domain is not None:
                obs_data_x, obs_data_y, mask = filter_obs_data(obs_data_x, obs_data_y, intervention_domain, ny_nodes)
            obs_data_x0, _ = label_pairs(D_obs, current_node, y_nodes)
            obs_data_x0 = obs_data_x0[mask]
            fy_pred = mean_fcn_fy(obs_data_x0)
            obs_data_fny = obs_data_y - fy_pred
            index_ini = tf.ones((1, len(ny_nodes)))

            # build the Gaussian Process Regression Model
            gprm, _, _ = build_gprm(
                index_x=index_ini,
                x=obs_data_x,
                y=obs_data_fny,
                obs_noise_factor=obs_noise_factor,
                learning_rate=learning_rate,
                max_training_step=max_training_step,
                debug_mode=debug_mode,
            )
            if debug_mode:
                plot_gprm_results(gprm, obs_data_x, obs_data_y)

            def mean_fcn(new_index, gprm=gprm):
                return gprm.get_marginal_distribution(new_index).mean()

            def std_fcn(new_index, gprm=gprm):
                return gprm.get_marginal_distribution(new_index).stddev()

            fny_fcns[t] = [mean_fcn, std_fcn]

        else:
            fy_fcns[t] = [None, None]

            assert len(ny_nodes) == len(predecessors)
            # if the target node has no predecessors at previous time steps
            # then the target node is impacted by a single emission functions
            obs_data_x, obs_data_y = label_pairs(D_obs, current_node, predecessors)
            if intervention_domain is not None:
                obs_data_x, obs_data_y, _ = filter_obs_data(obs_data_x, obs_data_y, intervention_domain, predecessors)
            index_ini = tf.ones((1, len(predecessors)))
            # build the Gaussian Process Regression Model
            gprm, _, _ = build_gprm(
                index_x=index_ini,
                x=obs_data_x,
                y=obs_data_y,
                obs_noise_factor=obs_noise_factor,
                learning_rate=learning_rate,
                max_training_step=max_training_step,
                debug_mode=debug_mode,
            )

            def mean_fcn(new_index, gprm=gprm):
                return gprm.get_marginal_distribution(new_index).mean()

            def std_fcn(new_index, gprm=gprm):
                return gprm.get_marginal_distribution(new_index).stddev()
            
            if debug_mode:
                plot_gprm_results(gprm, obs_data_x, obs_data_y)

            fny_fcns[t] = [mean_fcn, std_fcn]


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
                f[node] = (
                    lambda node: lambda sample, e_num: self.fcns[0][node](sample, e_num)
                )(node)
            return f

        def dynamic(self):
            f = OrderedDict()
            for node in self.sorted_nodes:
                f[node] = (
                    lambda node: lambda t, sample, e_num: self.fcns[t][node](
                        sample, e_num
                    )
                )(node)
            return f

    return SEMhat
