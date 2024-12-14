import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from utils.sequential_sampling import draw_samples_from_sem
from utils.costs import equal_cost
from utils.sem_estimate import sem_hat


class DynCausalBayesOpt:
    """
    Dynamic Causal Bayesian Optimization
    """

    def __init__(
        self,
        dyn_graph,
        sem,
        D_obs: OrderedDict,
        D_interven_ini: OrderedDict,
        intervention_domain: OrderedDict,
        num_trials: int,
        task: str = "min",
        cost_fcn: callable = equal_cost,
        num_anchor_points: int = 100,
    ):
        self.dyn_graph = dyn_graph
        self.sem = sem
        self.D_obs = D_obs
        self.D_interven_ini = D_interven_ini
        self.intervention_domain = intervention_domain
        self.num_trials = num_trials
        self.task = task
        self.T = len(self.dyn_graph.full_output_vars)
        self.D_interven = [D_interven_ini] + [OrderedDict() for _ in range(self.T - 1)]
        self.optimal_interventions = [[] for _ in range(self.T)]
        self.target_var = self.dyn_graph.full_output_vars[-1].split("_")[0]
        self.cost_fcn = cost_fcn
        self.sem_hat = OrderedDict()
        self.num_anchor_points = num_anchor_points
        self.exploration_set = self._initialize_exploration_set()

    def _initialize_exploration_set(self) -> list[list[str]]:
        self.dyn_graph.temporal_index = 0
        mis = self.dyn_graph.minimal_interven_set()
        # filter out the empty set
        exploration_set = [es for es in mis if es]
        # Create a new exploration set with modified node identifiers
        new_exploration_set = []
        for subset in exploration_set:
            # Create a new subset by taking the part before '_' in each node identifier
            new_subset = [node.split("_")[0] for node in subset]
            new_exploration_set.append(new_subset)
        return new_exploration_set

    def _optimal_observed_target(self, temporal_index: int) -> float:
        # get the minimal value in D_interven[temporal_index]
        if self.task == "min":
            return min(self.D_interven[temporal_index].values())
        elif self.task == "max":
            return max(self.D_interven[temporal_index].values())
        else:
            raise ValueError("Task should be either 'min' or 'max'.")

    def _pw_do_x_i(self):
        pass

    def _update_prior_causal_gp(self):
        pass

    def _posterior_causal_gp(self):
        pass

    def _acquisition_function(self):
        pass

    def run(self):
        for temporal_index in range(self.T):

            if temporal_index > 0:
                # Initialize dynamic causal GP models for all exploration sets
                # using the optimal intervention from the previous time step
                pass

            # Initialize the exploration set
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
