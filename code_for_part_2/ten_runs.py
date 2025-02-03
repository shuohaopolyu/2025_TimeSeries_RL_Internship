import sys

sys.path.append("..")
from sems import Y2X, U2XY, X2Y
from biocd import BIOCausalDiscovery
import tensorflow as tf
from collections import OrderedDict
import pickle

tf.random.set_seed(1111)

def run_experiment(model_class, max_pd_dc_limit=0.25):
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
    models = OrderedDict([ ("Y2X", Y2X), ("U2XY", U2XY), ("X2Y", X2Y) ])
    p_dc_limit = OrderedDict([ ("Y2X", 0.25), ("U2XY", 0.15), ("X2Y", 0.05) ])
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


if __name__ == "__main__":
    main()