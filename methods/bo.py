import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from utils.gaussian_process import build_gprm
from utils.costs import equal_cost
from utils.sequential_sampling import (
    draw_samples_from_sem_dev,
)
import networkx as nx


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
        cost_fcn: callable = equal_cost,
        num_anchor_points: int = 100,
    ):
        self.dyn_graph = dyn_graph
        self.sem = sem
        self.T = len(self.dyn_graph.full_output_vars)
        self.D_obs_full = OrderedDict(
            [(t, D_obs[key][:, t, None]) for key in D_obs.keys() for t in range(self.T)]
        )
        self.D_obs = D_obs
        self.obs_domain = obs_domain
        self.num_trials = num_trials
        self.task = task
        assert task in ["min", "max"], "Task should be either 'min' or 'max'."
        self.T = len(self.dyn_graph.full_output_vars)
        self.cost_fcn = cost_fcn
        self.num_anchor_points = num_anchor_points
        self.target_var = self.dyn_graph.full_output_vars[-1].split("_")[0]
        self.ancestors = self._ancestors_of_target_var()

    def run(self):
        """Run Bayesian Optimization"""
        for temporal_index in range(self.T):

            self._update_current_best_observation(temporal_index)

            for trial in range(self.num_trials):
                self._posterior_gp(temporal_index)

                self._acqusition_function()

                suspected_candidate = self._suspected_observation_this_trial()

                self._observe(temporal_index, suspected_candidate)

                self._update_current_best_observation(temporal_index)

                print("Temporal Index: ", temporal_index)
                print("Trial: ", trial)
                print("Candidate point observed in this trial: ", suspected_candidate)
                print("Observation: ", self.D_obs_full[temporal_index][self.target_var][-1, 0])
                print("Current Best Observation: ", self.y_star)

    def _ancestors_of_target_var(self) -> list:
        """Ancestors of Target Variable"""
        self.dyn_graph.temporal_index = 0
        the_graph = self.dyn_graph.graph.copy()
        target_var_at_0 = self.target_var + "_0"
        ancestors_at_0 = list(nx.ancestors(the_graph, target_var_at_0))
        self.ancestors = [x.split("_")[0] for x in ancestors_at_0]
        return self.ancestors

    def _posterior_gp(self, temporal_index) -> callable:
        """Posterior Gaussian Process"""
        i_D_obs = self.D_obs_full[temporal_index]
        ipt_x = None
        for key in self.D_obs.keys():
            if key == self.target_var:
                ipt_y = i_D_obs[key]
                continue
            if ipt_x is None:
                ipt_x = self.i_D_obs[key]
            else:
                ipt_x = tf.concat([ipt_x, self.i_D_obs[key]], axis=1)

        # Build GP
        index_x = ipt_x[0, None, :]
        self.gprm = build_gprm(index_x, ipt_x, ipt_y)
        return self.gprm

    def _update_current_best_observation(self, temporal_index) -> tf.Tensor:
        """Current Best Observation"""
        i_target_obs = self.D_obs_full[temporal_index][self.target_var]
        if self.task == "min":
            self.y_star = tf.reduce_min(i_target_obs)
            return self.y_star
        else:
            self.y_star = tf.reduce_max(i_target_obs)
            return self.y_star

    def _acqusition_function(self) -> tuple:
        """Acquisition Function"""
        candidate_points = self._candidate_points()
        mean_candidate = self.gprm.mean(candidate_points)[:, tf.newaxis]
        std_candidate = self.gprm.stddev(candidate_points)[:, tf.newaxis]
        _gaussians = tfp.distributions.Normal(
            loc=mean_candidate - self.y_star, scale=std_candidate
        )
        _gaussians_cdf = _gaussians.log_cdf(0.0)
        if self.task == "max":
            _gaussians_cdf = -_gaussians_cdf
        aquisition = _gaussians_cdf / self.cost_fcn(candidate_points)
        return candidate_points, aquisition

    def _suspected_observation_this_trial(self) -> tf.Tensor:
        """Suspected Observation This Trial"""
        candidate_points, acqusition = self._acqusition_function()
        suspected_candidate = candidate_points[tf.argmax(acqusition), :]
        return suspected_candidate

    def _candidate_points(self) -> tf.Tensor:
        """Candidate Points"""
        num_ancestors = len(self.ancestors)
        num_points = int(self.num_anchor_points ** (1 / num_ancestors))
        candidate_points = None
        for i, ancestor in enumerate(self.ancestors):
            domain = self.obs_domain[ancestor]
            if candidate_points is None:
                candidate_points = tfp.distributions.Uniform(
                    low=domain[0], high=domain[1]
                ).sample(num_points, 1)
            else:
                candidate_points = tf.concat(
                    [
                        candidate_points,
                        tfp.distributions.Uniform(low=domain[0], high=domain[1]).sample(
                            num_points, 1
                        ),
                    ],
                    axis=1,
                )
        return candidate_points

    def _observe(
        self,
        temporal_index: int,
        suspected_candidate: tf.Tensor,
        epsilon: OrderedDict | float = None,
    ) -> None:
        """Observe"""
        intervention = {}
        for i, key in enumerate(list(self.D_obs.keys())):
            intervention[key] = [None] * temporal_index
            if key == self.target_var:
                intervention[key].append(None)
            else:
                intervention[key].append(suspected_candidate[i])

        the_sample = draw_samples_from_sem_dev(
            self.sem, 1, temporal_index, intervention, epsilon
        )
        for key in the_sample.keys():
            self.D_obs_full[temporal_index] = tf.concat(
                [
                    self.D_obs_full[temporal_index],
                    the_sample[key][:, None, temporal_index],
                ],
                axis=1,
            )