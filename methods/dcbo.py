import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from utils.sequential_sampling import draw_samples_from_sem


class DynCausalBayesOpt:
    # not tested
    def __init__(self, dyn_graph, sem, num_trials: int, task: str= "min"):
        self.dyn_graph = dyn_graph
        self.sem = sem
        self.num_trials = num_trials
        self.task = task
        self.T = len(self.dyn_graph.full_output_vars)
        self.optimal_interventions = [[] for _ in range(self.T)]
        self.target_var = self.dyn_graph.full_output_vars[-1].split("_")[0]

    def _initialize_exploration_set(self, temporal_index) -> list[list[str]]:
        self.dyn_graph.temporal_index = temporal_index
        interven_hist = [[] for _ in range(self.T)]
        for i in range(temporal_index):
            if self.optimal_interventions[i] is not []:
                for optimal_intervention in self.optimal_interventions[i]:
                    interven_hist[i].append(optimal_intervention[0])
        self.dyn_graph.full_do_vars = interven_hist
        mis = self.dyn_graph.minimal_interven_set()
        delete_idx = []
        for i, es in enumerate(mis):
            for var in es:
                if var.split("_")[1] != str(temporal_index):
                    delete_idx.append(i)
                    break
        keep_idx = [i for i in range(len(mis)) if i not in delete_idx]
        exploration_set = [mis[i] for i in keep_idx]
        return exploration_set

    def optimize(self):
        for temporal_index in range(self.T):
            if temporal_index > 0:
                # Initialize dynamic causal GP models for all exploration sets
                # using the optimal intervention from the previous time step
                pass

            # Initialize the exploration set
            exploration_set = self._initialize_exploration_set(temporal_index)

            for trial in range(self.num_trials):
                # Compute acquisition function for each exploration set
                pass
                # Obtain the optimal intervention among all exploration sets
                pass
                # Intervene
                pass
                # Agument the intervention history
                pass
                # Update posterior for each GP model to predict the optimal target variable
                pass

            # Obtain the optimal intervention for the current time step
            pass

            # Update the optimal intervention history
            pass


