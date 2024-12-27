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


dyn_graph = three_step_stat()
sem_model = StationaryModel_dev()
full_samples = OrderedDict([(key, []) for key in sem_model.static().keys()])
num_samples = 20
max_time_step = 2
epsilon = OrderedDict(
    [
        (key, tfd.Normal(0.0, 1.0).sample((num_samples, max_time_step)))
        for key in full_samples.keys()
    ]
)

D_obs = draw_samples_from_sem_dev(sem_model, num_samples, max_time_step, seed=1111)
intervention_ini = {
    "X": [-3.6],
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
intervention_domain = OrderedDict([("X", [-5.0, 5.0]), ("Z", [-5.0, 20.0])])
num_trials = 20
task = "min"
cost_fn = equal_cost
num_anchor_points = 100
num_monte_carlo = 10
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

dcbo._update_observational_data(0)
dcbo._update_sem_hat(0)
dcbo._prior_causal_gp(0)

z_mean_fcn, z_std_fcn = dcbo.prior_causal_gp[("Z",)]
x_mean_fcn, x_std_fcn = dcbo.prior_causal_gp[("X",)]

z_index = (tf.linspace(-5.0, 20.0, 100))[:, tf.newaxis]
x_index = (tf.linspace(-5.0, 5.0, 100))[:, tf.newaxis]

# z_mean = z_mean_fcn(z_index)
# z_std = z_std_fcn(z_index)
# print(z_std)
# z_true = tf.cos(z_index) - tf.exp(-z_index / 20.0)

x_mean = x_mean_fcn(x_index)
x_std = x_std_fcn(x_index)
x2z_true = tf.exp(-x_index)
x_true = tf.cos(x2z_true) - tf.exp(-x2z_true / 20.0)


# plt.plot(z_index.numpy(), z_mean.numpy())
# plt.plot(z_index.numpy(), z_true.numpy(), color="k")
# plt.scatter(D_obs["Z"][:, 0], D_obs["Y"][:, 0], color="r", marker="x")
# plt.fill_between(
#     z_index[:, 0].numpy(),
#     z_mean.numpy() - 2 * z_std.numpy(),
#     z_mean.numpy() + 2 * z_std.numpy(),
#     alpha=0.4,
#     color="k",
# )
# plt.show()

plt.plot(x_index.numpy(), x_mean.numpy())
plt.plot(x_index.numpy(), x_true.numpy(), color="k")
plt.scatter(D_obs["X"][:, 0], D_obs["Y"][:, 0], color="r", marker="x")
plt.fill_between(
    x_index[:, 0].numpy(),
    x_mean.numpy() - 2 * x_std.numpy(),
    x_mean.numpy() + 2 * x_std.numpy(),
    alpha=0.4,
    color="k",
)
plt.show()

