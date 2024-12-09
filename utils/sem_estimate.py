from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp


def fit_arcs(dyn_graph: object, D_obs: OrderedDict, emissions: bool) -> dict:
    """Each edge is modeled as a Gaussian Process, 
    and the hyperparameters are optimized using the observation data.

    Args:
        dyn_graph (object): dynamic graph object
        D_obs (OrderedDict): observation dataset
        emissions (bool): if True, the function will return the emission functions

    Returns:
        OrderedDict: contains callable functions that represent the edges
    """
    fcns = {t: {} for t in range(len(dyn_graph.full_output_vars))}
    pass


# given the D_obs, return approximated scm model
def sem_hat(dyn_graph, emission_fcns, transition_fncs):
    pass
