import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from utils.sequential_sampling import draw_samples_from_sem, draw_samples_from_sem_hat
from utils.costs import equal_cost
from utils.sem_estimate import sem_hat, fy_and_fny


class DynCausalBayesOpt:
    """Dynamic Causal Bayesian Optimization

    Args:
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
        general_exploration_set = []
        for subset in exploration_set:
            # Create a new subset by taking the part before '_' in each node identifier
            new_subset = [node.split("_")[0] for node in subset]
            general_exploration_set.append(new_subset)
        return general_exploration_set

    def _optimal_interven_key(self, temporal_index: int) -> float:
        # get the minimal value in D_interven[temporal_index]
        assert temporal_index > 0, "temporal_index should be greater than 0."
        if self.task == "min":
            return min(
                self.D_interven[temporal_index - 1],
                key=lambda k: self.D_interven[temporal_index - 1][k],
            )
        elif self.task == "max":
            return max(
                self.D_interven[temporal_index - 1],
                key=lambda k: self.D_interven[temporal_index - 1][k],
            )
        else:
            raise ValueError("Task should be either 'min' or 'max'.")

    def intervene_scheme(self):
        pass

    def _causal_gp_prior(
        self, temporal_index: int, num_monte_carlo: int
    ) -> OrderedDict:
        # Initialize the prior causal GP model for each exploration set
        # returen an OrderedDict, where the key is the exploration set and the value is a tuple
        # containing the callable prior mean and covariance functions
        prior_gp = OrderedDict()
        self.dyn_graph.temporal_index = temporal_index
        the_graph = self.dyn_graph.graph
        fy_fcn, fny_fcn = fy_and_fny(
            the_graph, self.D_obs, self.target_var, temporal_index
        )
        for es in self.exploration_set:
            # Initialize the prior mean and covariance functions for each exploration set
            if fy_fcn[0] is not None:
                opt_interven_key = self._optimal_interven_key(temporal_index)
                opt_interven_val, _ = self.D_interven[temporal_index - 1][
                    opt_interven_key
                ]
                optimal_intervene_scheme = self.intervene_scheme()
                samples = draw_samples_from_sem_hat(
                    self.sem_hat,
                    num_monte_carlo,
                    max_time_step=temporal_index,
                    intervention=optimal_intervene_scheme,
                )
                sampled_f_star = samples[self.target_var][:, temporal_index-1]
                # compute the mean of f_star
                mean_f_star = tf.reshape(tf.reduce_mean(sampled_f_star), [1, 1])
                # compute the mean of fy(f_star)
                mean_fy_f_star = fy_fcn[0](mean_f_star)
                std_fy_f_star = fy_fcn[1](mean_f_star)

            prior_gp[es] = [None, None]
        return prior_gp

    def _posterior_causal_gp(self):
        pass

    def _acquisition_function(self):
        pass

    def run(self):
        for temporal_index in range(self.T):

            self._prior_causal_gp()

            if temporal_index > 0:
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
