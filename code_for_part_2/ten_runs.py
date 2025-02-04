import sys

sys.path.append("..")
from sems import Y2X, U2XY, X2Y
from biocd import BIOCausalDiscovery
import tensorflow as tf
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
tf.random.set_seed(1111)



def run_experiment(model_class, max_pd_dc_limit=0.00):
    model = model_class()
    D_obs = model.propagate(5000)
    cd = BIOCausalDiscovery(
        true_sem=model,
        D_obs=D_obs,
        D_int=OrderedDict(),
        num_int=10,
        k_0=10.0,
        k_1=0.1,
        beta=0.2,
        num_monte_carlo=4096,
        num_mixture=50,
        num_samples_per_sem=100000,
        intervention_domain=None,
        max_iter=20000,
        learning_rate=0.001,
        patience=20,
        debug_mode=False,
    )
    ini_max_p_dc = cd._ini_max_p_dc()
    if ini_max_p_dc < max_pd_dc_limit:
        return None

    cd.run()
    result = cd.get_recorder_data()
    return result

def main():
    models = OrderedDict([("Y2X", Y2X), ("U2XY", U2XY), ("X2Y", X2Y)])
    p_dc_limit = OrderedDict([("Y2X", 0.10), ("U2XY", 0.10), ("X2Y", 0.10)])
    replicate_times = 10
    for model_name, model in models.items():
        results = []
        run_times = 0
        while run_times < replicate_times:
            print(f"Running {model_name} {run_times+1}/{replicate_times} ...")
            result = run_experiment(model, p_dc_limit[model_name])
            if result is not None:
                results.append(result)
                run_times += 1
            else:
                print(f"Observation data is not good enough for causal discovery in {model_name}, retrying ...")

        filename = f"./exps/{model_name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved to {filename}")

def plot_interval(ax_obj, x, y):
    x = np.array(x)
    x = np.reshape(x, (-1))
    std = np.std(y, axis =1)
    mean = np.mean(y, axis=1)
    ax_obj.fill_between(x, mean - std, mean + std, alpha=0.3)
    ax_obj.plot(x, mean, linestyle="--", color="black")


def plot_results():

    from matplotlib import rcParams
    plt.rcParams.update({"font.size": 8})
    cm = 1 / 2.54 
    config = {
        "font.weight": "normal",
        "font.size": 8,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    }
    rcParams.update(config)
    plt.rcParams["axes.unicode_minus"] = True

    models = OrderedDict([("Y2X", Y2X), ("U2XY", U2XY), ("X2Y", X2Y)])
    fig, axs = plt.subplots(3, 4, figsize=(16*cm, 12*cm))
    x = range(10)
    for j, (model_name, model) in enumerate(models.items()):
        filename = f"./exps/{model_name}.pkl"
        pdc = np.zeros((10, 10))
        bf = np.zeros((10, 10))
        p_h0 = np.zeros((10, 10))
        p_h1 = np.zeros((10, 10))
        with open(filename, "rb") as f:
            results = pickle.load(f)
        for i, result in enumerate(results):
            pdc[:, i] = result["max_p_dc"]
            bf[:, i] = result["bayes_factor_01_int"]
            p_h0[:, i] = result["p_h0_D_int"]
            p_h1[:, i] = result["p_h1_D_int"]
        plot_interval(axs[j, 0], x, pdc)
        plot_interval(axs[j, 1], x, bf)
        plot_interval(axs[j, 2], x, p_h0)
        plot_interval(axs[j, 3], x, p_h1)

        
    ylabels = [r"$P_{DC}$", r"$\log \text{BF}_{01}$", r"$P({\mathbb{H}_0} | \text{D}_{\text{int}})$", r"$P(\mathbb{H}_1 | \text{D}_{\text{int}})$"]
    titles = [r"$\mathbb{H}_0: X \leftarrow Y$", r"$\mathbb{H}_0: X \leftarrow U \rightarrow Y$", r"$\mathbb{H}_1: X \rightarrow Y$"]
    for i in range(4):
        for j in range(3):
            axs[j, i].set_ylabel(ylabels[i], fontsize=8, labelpad=0)
            axs[j, i].set_xlabel("Active sampling size", fontsize=8, labelpad=0)
            axs[j, i].set_xlim([0, 9])
            axs[j, i].set_xticks([0, 3, 6, 9])
            axs[j, i].tick_params(axis="both", direction="in")
            axs[j, i].set_title(titles[j], fontsize=8)
            axs[j, 0].set_ylim([0.0, 1.1])
            axs[j, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
            axs[j, i].text(-0.2, -0.2, "(" + chr(97+j*4+i) + ")", transform=axs[j, i].transAxes)
    axs[0, 1].set_ylim([0, 3.6])
    axs[0, 1].set_yticks([0, 1.2, 2.4, 3.6], ["0", "1.2", "2.4", "3.6"])
    axs[1, 1].set_ylim([0, 4.8])
    axs[1, 1].set_yticks([0, 1.6, 3.2, 4.8], ["0", "1.6", "3.2", "4.8"])
    axs[2, 1].set_ylim([-3.6, 0])
    axs[2, 1].set_yticks([-3.6, -2.4, -1.2, 0], ["-3.6", "-2.4", "-1.2", "0"])
    axs[0, 2].set_ylim([0.5, 1.0])
    axs[0, 2].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    axs[1, 2].set_ylim([0.5, 1.0])
    axs[1, 2].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    axs[2, 2].set_ylim([0.0, 0.5])
    axs[2, 2].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    axs[0, 3].set_ylim([0.0, 0.5])
    axs[0, 3].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    axs[1, 3].set_ylim([0.0, 0.5])
    axs[1, 3].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    axs[2, 3].set_ylim([0.5, 1.0])
    axs[2, 3].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "0.6", "0.7", "0.8", "0.9", "1"])

    plt.tight_layout(pad=0.1)
    plt.savefig("./exps/ten_runs.pdf")
    plt.show()


if __name__ == "__main__":
    # main()
    plot_results()