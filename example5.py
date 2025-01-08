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
tf.random.set_seed(1)



def plot_prior_x():
    sem_model = StationaryModel_dev()

    num_samples = 80
    temporal_index = 2
    full_samples = OrderedDict([(key, []) for key in sem_model.static().keys()])
    epsilon = OrderedDict(
        [
            (key, tfd.Normal(0.0, 1.0).sample((num_samples, temporal_index+1)))
            for key in full_samples.keys()
        ]
    )
    D_obs = draw_samples_from_sem_dev(sem_model, num_samples, temporal_index, epsilon=epsilon)
    print(D_obs)

    dyn_graph = three_step_stat()

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
    num_monte_carlo = 20
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
        ini_global_extreme_abs=1e3,
        learning_rate=1e-4,
        intervene_noise_factor=1e-4,
        observation_noise_factor=1e-2,
        max_training_step=20000,
        debug_mode=False,
    )

    opt_history = [[] for _ in range(dcbo.T)]
    suspected_es = None
    dcbo._update_observational_data(1)
    dcbo._update_sem_hat(1)
    dcbo.opt_intervene_history[0] = OrderedDict()
    dcbo.opt_intervene_history[0]["decision_vars"] = (("Z",), [-3.2])
    dcbo.opt_intervene_history[0]["optimal_value"] = -2.1
    priors = dcbo._prior_causal_gp(1)


    def y_x(x_samples):
        z_samples = tf.exp(-x_samples)-3.2
        return -tf.exp(-z_samples/20.0) + tf.cos(z_samples) - 2.1

    candidate_points = tf.linspace(-5.0, 5.0, 300)[:, tf.newaxis]
    x_prior_mean, x_prior_std = priors[("X",)]
    pred_mean = x_prior_mean(candidate_points)
    pred_std = x_prior_std(candidate_points)


    ref_y = y_x(candidate_points)

    centimeters = 1 / 2.54
    plt.figure(figsize=(11*centimeters, 5.5*centimeters))
    plt.plot(candidate_points[:, 0], ref_y[:, 0], color="red", linestyle="--", label="Ref.")
    plt.plot(candidate_points[:, 0], pred_mean, label="Prior mean")

    plt.fill_between(
        candidate_points[:, 0],
        pred_mean - 2 * pred_std,
        pred_mean + 2 * pred_std,
        alpha=0.2,
        label="Prior 95% CI"
    )
    plt.legend(fontsize=8, loc="upper right", ncol=3)
    plt.xlim(-5, 5)
    plt.xticks(tf.linspace(-5, 5, 6))
    plt.ylim(-8, 4)
    plt.yticks(tf.linspace(-8, 4, 7))
    plt.xlabel("$X_1$", fontsize=8)
    plt.ylabel("$Y_1$", fontsize=8)
    plt.tick_params(axis='both', direction='in', labelsize=8)
    plt.tight_layout(pad=0.1)
    # plt.savefig("./demo/experiments/prior_x1.pdf")
    plt.show()

if __name__ == "__main__":
    plot_prior_x()

