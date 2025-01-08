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
# tf.random.set_seed(1)

sem_model = StationaryModel_dev()
num_samples = 100
temporal_index = 2
full_samples = OrderedDict([(key, []) for key in sem_model.static().keys()])
epsilon = OrderedDict(
    [
        (key, tfd.Normal(0.0, 0.1).sample((num_samples, temporal_index + 1)))
        for key in full_samples.keys()
    ]
)

epsilon_x1 = tf.linspace(-5.0, 5.0, num_samples)[:, tf.newaxis]
epsilon_x23 = tfd.Normal(0.0, 0.1).sample((num_samples, 2))
epsilon["X"] = tf.concat([epsilon_x1, epsilon_x23], axis=1)
D_obs = draw_samples_from_sem_dev(
    sem_model, num_samples, temporal_index, epsilon=epsilon
)

dyn_graph = three_step_stat()
# D_obs = draw_samples_from_sem_dev(sem_model, num_samples, temporal_index)

intervention_ini = {
    "X": [-0.6],
    "Z": [None],
    "Y": [None],
}
D_intervene_ini_x1 = draw_samples_from_sem_dev(
    sem_model, 1, 0, intervention=intervention_ini, epsilon=0.0
)

intervention_ini = {
    "X": [1.2],
    "Z": [None],
    "Y": [None],
}
D_intervene_ini_x2 = draw_samples_from_sem_dev(
    sem_model, 1, 0, intervention=intervention_ini, epsilon=0.0
)

D_intervene_ini_x = OrderedDict()
for key in D_intervene_ini_x1.keys():
    D_intervene_ini_x[key] = tf.concat(
        [D_intervene_ini_x1[key], D_intervene_ini_x2[key]], axis=0
    )

intervention_ini = {
    "X": [None],
    "Z": [2.5],
    "Y": [None],
}
D_intervene_ini_z = draw_samples_from_sem_dev(
    sem_model, 1, 0, intervention=intervention_ini, epsilon=0.0
)
D_intervene_ini = OrderedDict(
    [(("X",), D_intervene_ini_x), (("Z",), D_intervene_ini_z)]
)
# print(D_intervene_ini)
intervention_domain = OrderedDict([("X", [-5.0, 5.0]), ("Z", [-5.0, 20.0])])
num_trials = 15
task = "min"
cost_fn = equal_cost
num_anchor_points = 100
num_monte_carlo = 50
jitter = 1e-6
dcbo = DynCausalBayesOpt(
    dyn_graph,
    sem_model,
    D_obs,
    D_intervene_ini,
    intervention_domain,
    num_trials,
    task,
    cost_fn,
    num_anchor_points,
    num_monte_carlo,
    jitter,
)

opt_history = [[] for _ in range(dcbo.T)]
suspected_es = None

dcbo._update_observational_data(1)
dcbo._update_sem_hat(1)
dcbo.opt_intervene_history[0] = OrderedDict()
dcbo.opt_intervene_history[0]["decision_vars"] = (("Z",), [tf.constant(-3.2)])
dcbo.opt_intervene_history[0]["optimal_value"] = -2.1
priors = dcbo._prior_causal_gp(1)

candidate_points = tf.linspace(-5.0, 20.0, 100)[:, tf.newaxis]

x_prior_mean, x_prior_std = priors[("Z",)]

pred_mean = x_prior_mean(candidate_points)
pred_std = x_prior_std(candidate_points)

plt.plot(candidate_points, pred_mean, label="prior mean")
plt.fill_between(
    tf.squeeze(candidate_points),
    tf.squeeze(pred_mean - 2 * pred_std),
    tf.squeeze(pred_mean + 2 * pred_std),
    alpha=0.2,
    label="prior std",
)
plt.show()


