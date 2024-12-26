import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict


class BayesOpt:
    """Bayesian Optimization"""

    def __init__(
        self,
        dyn_graph: callable,
        sem: callable,
        D_obs: OrderedDict,
        obs_domain: OrderedDict,
        num_trials: int,
        task: str,
        cost_fcn: callable,
        num_anchor_points: int = 100,
    ):
        self.dyn_graph = dyn_graph
        self.sem = sem
        self.D_obs = D_obs
        self.obs_domain = obs_domain
        self.num_trials = num_trials
        self.task = task
        assert task in ["min", "max"], "Task should be either 'min' or 'max'."
        self.T = len(self.dyn_graph.full_output_vars)
        self.cost_fcn = cost_fcn
        self.num_anchor_points = num_anchor_points
        self.target_var = self.dyn_graph.full_output_vars[-1].split("_")[0]

    def run(self):
        """Run Bayesian Optimization"""
        for temporal_index in range(self.T):

            for trial in range(self.num_trials):
                self._posterior_gp(temporal_index)

                self._acqusition_function(temporal_index)

                suspected_candidate = self._suspected_observation_this_trial()

                self._observe(suspected_candidate)

                self._update_opt_observation(temporal_index)

    def _posterior_gp(self, temporal_index):
        """Posterior Gaussian Process"""
        pass

    def _acqusition_function(self, temporal_index):
        """Acquisition Function"""
        pass

    def _suspected_observation_this_trial(self):
        """Suspected Observation This Trial"""
        pass

    def _observe(self, suspected_candidate):
        """Observe"""
        pass

    def _update_opt_observation(self, temporal_index):
        """Update Optimal Observation"""
        pass
