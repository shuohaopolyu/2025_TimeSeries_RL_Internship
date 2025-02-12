import sys
sys.path.append("..")

from methods.dcbo import DynCausalBayesOpt
from utils.sequential_sampling import draw_samples_from_sem_dev
from causal_graph.example_dyn_graphs import three_step_stat
from sem.stationary import StationaryModel_dev
from collections import OrderedDict
from utils.costs import equal_cost
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
tfd = tfp.distributions

sem_model = StationaryModel_dev()
num_samples = 100
temporal_index = 2
full_samples = OrderedDict([(key, []) for key in sem_model.static().keys()])
epsilon = OrderedDict(
    [
        (key, tfd.Normal(0.0, 0.1).sample((num_samples, temporal_index+1)))
        for key in full_samples.keys()
    ]
)
epsilon_x1 = tf.linspace(-5.0, 5.0, num_samples)[:, tf.newaxis]
epsilon_x23 = tfd.Normal(0.0, 0.1).sample((num_samples, 2))
epsilon["X"] = tf.concat([epsilon_x1, epsilon_x23], axis=1)
D_obs = draw_samples_from_sem_dev(sem_model, num_samples, temporal_index, epsilon=epsilon)

dyn_graph = three_step_stat()

intervention_ini = {
    "X": [-0.6],
    "Z": [None],
    "Y": [None],
}
D_intervene_ini_x = draw_samples_from_sem_dev(
    sem_model, 1, 0, intervention=intervention_ini, epsilon=0.0
)

intervention_ini = {
    "X": [None],
    "Z": [7.5],
    "Y": [None],
}
D_intervene_ini_z = draw_samples_from_sem_dev(
    sem_model, 1, 0, intervention=intervention_ini, epsilon=0.0
)
D_intervene_ini = OrderedDict(
    [(("X",), D_intervene_ini_x), (("Z",), D_intervene_ini_z)]
)
intervention_domain = OrderedDict([("X", [-3.0, 5.0]), ("Z", [-5.0, 20.0])])
dcbo = DynCausalBayesOpt(
    dyn_graph=dyn_graph,
    sem=sem_model,
    D_obs=D_obs,
    D_intervene_ini=D_intervene_ini,
    intervention_domain=intervention_domain,
    num_trials=10,
    task="min",
    cost_fcn=equal_cost,
    num_anchor_points= 100,
    num_monte_carlo= 100,
    ini_global_extreme_abs=10.0,
    jitter= 1e-6,
    learning_rate=1e-4,
    intervene_noise_factor=1e-2,
    observation_noise_factor=1.0,
    max_training_step=100000,
    debug_mode=False,
)

opt_history = dcbo.run()