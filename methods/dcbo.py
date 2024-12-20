import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from utils.sequential_sampling import draw_samples_from_sem, draw_samples_from_sem_hat
from utils.costs import equal_cost
from utils.sem_estimate import sem_hat, fy_and_fny, label_pairs, fcns4sem
from utils.gaussian_process import build_gprm


class DynCausalBayesOpt:
    """Dynamic Causal Bayesian Optimization

    Args:

    dyn_graph (callable): The dynamic graph object.
    sem (callable): The structural equation model object.
    D_obs (OrderedDict): The observation data.
    D_intervene_ini (OrderedDict): The initial intervention data.
    intervention_domain (OrderedDict): The domain of each intervention variable.
    num_trials (int): The number of trials for each time step.
    task (str): The optimization task, either 'min' or 'max'.
    cost_fcn (callable): The cost function.
    num_anchor_points (int): The number of anchor points for each exploration set.
    num_monte_carlo (int): The number of Monte Carlo samples.
    jitter (float): The jitter that is added to the suspected intervention point.
    """

    def __init__(
        self,
        dyn_graph: callable,
        sem: callable,
        D_obs: OrderedDict,
        D_intervene_ini: OrderedDict,
        intervention_domain: OrderedDict,
        num_trials: int,
        task: str = "min",
        cost_fcn: callable = equal_cost,
        num_anchor_points: int = 100,
        num_monte_carlo: int = 10000,
        jitter: float = 1e-6,
    ):
        self.dyn_graph = dyn_graph
        self.sem = sem
        self.D_obs = D_obs
        self.intervention_domain = intervention_domain
        self.num_trials = num_trials
        self.task = task
        assert task in ["min", "max"], "Task should be either 'min' or 'max'."
        self.T = len(self.dyn_graph.full_output_vars)
        self.D_interven = OrderedDict()
        self.D_interven[0] = D_intervene_ini
        self.opt_intervene_history = OrderedDict()
        self.full_opt_intervene_vars = []
        self.target_var = self.dyn_graph.full_output_vars[-1].split("_")[0]
        self.cost_fcn = cost_fcn
        self.sem_hat = OrderedDict()
        self.num_anchor_points = num_anchor_points
        self.exploration_set = self._initialize_exploration_set()
        self.num_monte_carlo = num_monte_carlo
        self.prior_causal_gp = OrderedDict()
        self.posterior_causal_gp = OrderedDict()
        self.causal_gpm_list = OrderedDict()
        self.jitter = jitter

    def run(self):
        for temporal_index in range(self.T):
            # Update the observational data
            self._update_observational_data(temporal_index)

            # Update the estimated SEM model
            self._update_sem_hat(temporal_index)

            # nitialise dynamic causal GP models
            self._prior_causal_gp(temporal_index)

            # Initialize the exploration set
            for trial in range(self.num_trials):
                # Compute acquisition function for each exploration set
                self._acquisition_function(temporal_index)
                # Obtain the suspected optimal intervention based on the acquisition function
                suspected_es, suspected_candidate_point = (
                    self._suspected_intervention_this_trial()
                )
                # Intervene and augment the intervention history
                self._intervene_and_augment(
                    temporal_index, suspected_es, suspected_candidate_point
                )
                # Update posterior for each GP model to predict the optimal target variable
                self._posterior_causal_gp(temporal_index)

            # Update the optimal intervention history
            self._update_opt_intervene_history(temporal_index)

    def _initialize_exploration_set(self) -> list[list[str]]:
        self.dyn_graph.temporal_index = 0
        mis = self.dyn_graph.minimal_intervene_set()
        # filter out the empty set
        exploration_set = [es for es in mis if es]
        # Create a new exploration set with modified node identifiers
        general_exploration_set = []
        for subset in exploration_set:
            # Create a new subset by taking the part before '_' in each node identifier
            new_subset = [node.split("_")[0] for node in subset]
            general_exploration_set.append(new_subset)
        return general_exploration_set

    def _optimal_intervene_value(
        self, temporal_index: int
    ) -> tuple[tuple[str], list, float]:
        i_D_interven = self.D_interven[temporal_index]
        extreme_values = []
        for _, sub_dict in i_D_interven.items():
            target_tensor = sub_dict[self.target_var]
            sliced_tensor = target_tensor[:, temporal_index]
            if self.task == "min":
                extreme_values.append(tf.reduce_min(sliced_tensor))
            elif self.task == "max":
                extreme_values.append(tf.reduce_max(sliced_tensor))

        es_values = []
        if self.task == "min":
            global_extreme = tf.reduce_min(extreme_values).numpy()
            es_index = tf.keras.backend.eval(tf.argmin(extreme_values))
            es = tuple(i_D_interven.keys())[es_index]
            if type(es) == str:
                es = es
            global_extreme_index = tf.keras.backend.eval(
                tf.argmin(i_D_interven[es][self.target_var][:, temporal_index])
            )

        elif self.task == "max":
            global_extreme = tf.reduce_max(extreme_values).numpy()
            es_index = tf.keras.backend.eval(tf.argmax(extreme_values))
            es = tuple(i_D_interven.keys())[es_index]
            if type(es) == str:
                es = es
            global_extreme_index = tf.keras.backend.eval(
                tf.argmax(i_D_interven[es][self.target_var][:, temporal_index])
            )

        for key in es:
            es_values.append(
                i_D_interven[es][key][global_extreme_index, temporal_index].numpy()
            )

        return es, es_values, global_extreme

    def _intervene_scheme(
        self, x_py: list[str], i_py: list[str], x_py_values: list
    ) -> dict:
        assert len(x_py) == len(
            x_py_values
        ), "Length mismatch between x_py and x_py_values."
        max_time_step = int(x_py[0].split("_")[1]) + 1
        sem_keys = list(self.sem.static().keys())
        intervention = {key: [None] * max_time_step for key in sem_keys}
        for node in i_py:
            key = node.split("_")[0]
            node_index = int(node.split("_")[1])
            opt_es, opt_es_values = self.opt_intervene_history[node_index][
                "decision_vars"
            ]
            intervention[key][node_index] = opt_es_values[opt_es.index(key)]
        for node, value in zip(x_py, x_py_values):
            key = node.split("_")[0]
            node_index = int(node.split("_")[1])
            intervention[key][node_index] = value
        return intervention

    def _update_opt_intervene_vars(self, temporal_index: int) -> None:
        opt_es = self.opt_intervene_history[temporal_index]["decision_vars"][0]
        for node in opt_es:
            self.full_opt_intervene_vars.append(node+"_{}".format(temporal_index))

    def _update_opt_intervene_history(self, temporal_index: int) -> None:
        es, es_values, global_extreme = self._optimal_intervene_value(temporal_index)
        self.opt_intervene_history[temporal_index] = {
            "decision_vars": (es, es_values),
            "optimal_value": global_extreme,
        }

    def _input_fny(self, the_graph: object, samples: tf.Tensor) -> tf.Tensor:
        temporal_index = samples[self.target_var].shape[1] - 1
        current_node = self.target_var + "_" + str(temporal_index)
        predecessors = list(the_graph.predecessors(current_node))
        ny_nodes = []
        for predecessor in predecessors:
            predecessor_index = int(predecessor.split("_")[1])
            if predecessor_index == temporal_index:
                ny_nodes.append(predecessor)
        return label_pairs(samples, current_node, ny_nodes)[0]

    def _prior_causal_gp(self, temporal_index: int) -> OrderedDict:
        # Initialize the prior causal GP model for each exploration set
        # returen an OrderedDict, where the key is the exploration set and the value is a tuple
        # containing the callable prior mean and covariance functions
        self.dyn_graph.temporal_index = temporal_index
        the_graph = self.dyn_graph.graph
        predecessor_of_target = list(
            the_graph.predecessors(self.target_var + "_" + str(temporal_index))
        )
        fy_fcn, fny_fcn = fy_and_fny(
            the_graph, self.D_obs, self.target_var, temporal_index
        )
        for es in self.exploration_set:
            predecessor = predecessor_of_target.copy()
            # Initialize the prior mean and covariance functions for each exploration set
            if fy_fcn[0] is not None:
                # f_star = self._optimal_intervene_value(temporal_index), this could be a substitute to the following line
                f_star = self.opt_intervene_history[temporal_index]["optimal_value"]
                # compute the mean of fy(f_star)
                mean_fy_f_star = fy_fcn[0](f_star)
                std_fy_f_star = fy_fcn[1](f_star)
                # exclude the y_pt from the predecessor
                predecessor.remove(self.target_var + "_" + str(temporal_index - 1))
                samples_fy_f_star = tfp.distributions.Normal(
                    mean_fy_f_star, std_fy_f_star
                ).sample(self.num_monte_carlo)

            x_py = [x + "_" + str(temporal_index) for x in es]
            # get the subset of the predecessor that subscript is less than temporal_index
            previous_py = [
                node for node in predecessor if int(node.split("_")[1]) < temporal_index
            ]
            i_py = [
                node
                for node in previous_py
                if node.split("_")[0] in self.self.full_opt_intervene_vars
            ]

            def mean_fny_xiw(x_py_values):
                intervention = self._intervene_scheme(x_py, i_py, x_py_values)
                samples = draw_samples_from_sem_hat(
                    self.sem_estimated,
                    self.num_monte_carlo,
                    temporal_index + 1,
                    intervention=intervention,
                )
                input_fny = self._input_fny(samples)
                samples_mean_fny_xiw = fny_fcn[0](input_fny)
                return samples_mean_fny_xiw

            def prior_mean(x_py_values):
                samples_mean_fny_xiw = mean_fny_xiw(x_py_values)
                if fy_fcn[0] is not None:
                    return tf.reduce_mean(samples_mean_fny_xiw + samples_fy_f_star)
                else:
                    return tf.reduce_mean(samples_mean_fny_xiw)

            def prior_std(x_py_values):
                samples_mean_fny_xiw = mean_fny_xiw(x_py_values)
                if fy_fcn[0] is not None:
                    return tf.math.reduce_std(samples_mean_fny_xiw + samples_fy_f_star)
                else:
                    return tf.math.reduce_std(samples_mean_fny_xiw)

            self.prior_causal_gp[es] = (prior_mean, prior_std)
        return self.prior_causal_gp

    def _posterior_causal_gp(self, temporal_index: int) -> OrderedDict:
        for es in self.exploration_set:
            if self.D_interven[temporal_index] is None:
                # No intervention is performed
                self.posterior_causal_gp[es] = self.prior_causal_gp[es]
            else:
                # Intervention is performed
                es_interven = self.D_interven[temporal_index][es]
                es_intervene_x = []
                for key in es:
                    es_intervene_x.append(es_interven[key][:, temporal_index])
                es_intervene_x = tf.convert_to_tensor(es_intervene_x)
                es_intervene_y = es_interven[self.target_var][:, temporal_index]
                index_x = tf.reshape(es_intervene_x[0, :] + self.jitter, [1, -1])
                causal_gpm, _, _ = build_gprm(
                    index_x,
                    es_intervene_x,
                    es_intervene_y,
                    mean_fn=self.prior_causal_gp[es][0],
                    causal_std_fn=self.prior_causal_gp[es][1],
                    debug_mode=True,
                    obs_noise_factor=0.0,
                    amplitude_factor=(
                        self.causal_gpm_list[es].kernel.amplitude
                        if self.causal_gpm_list[es] is not None
                        else 1.0
                    ),
                    length_scale_factor=(
                        self.causal_gpm_list[es].kernel.length_scale
                        if self.causal_gpm_list[es] is not None
                        else 1.0
                    ),
                )

            def posterior_mean(x_py_values):
                return causal_gpm.get_margin_distribution(x_py_values).mean()

            def posterior_std(x_py_values):
                return causal_gpm.get_margin_distribution(x_py_values).stddev()

            self.posterior_causal_gp[es] = (posterior_mean, posterior_std)

        return self.posterior_causal_gp

    def _intervention_points(self, es) -> tf.Tensor:
        # Generate the candidate points for each exploration set within the intervention domain
        # return the candidate points
        if len(es) == 1:
            intervention_min, intervention_max = self.intervention_domain[es[0]]
            candidate_points = tf.linspace(
                intervention_min, intervention_max, self.num_anchor_points
            )[:, tf.newaxis]
        else:
            for i, node in enumerate(es):
                if i == 0:
                    intervention_min, intervention_max = self.intervention_domain[node]
                    candidate_points = tf.linspace(
                        intervention_min, intervention_max, self.num_anchor_points
                    )[:, tf.newaxis]
                else:
                    intervention_min, intervention_max = self.intervention_domain[node]
                    candidate_points = tf.concat(
                        [
                            candidate_points,
                            tf.linspace(
                                intervention_min,
                                intervention_max,
                                self.num_anchor_points,
                            )[:, tf.newaxis],
                        ],
                        axis=1,
                    )
        return candidate_points

    def _acquisition_function(self, temporal_index: int) -> OrderedDict:
        self.D_acquisition = OrderedDict()
        for es in self.exploration_set:
            candidate_points = self._intervention_points(es)
            self.D_acquisition[es][0] = candidate_points
            _, _, y_star = self._optimal_intervene_value(temporal_index)
            posterior_mean_candidate_points = self.posterior_causal_gp[es][0](
                candidate_points
            )
            posterior_std_candidate_points = self.posterior_causal_gp[es][1](
                candidate_points
            )
            if self.task == "min":
                truncated_gaussian = tfp.distributions.TruncatedNormal(
                    posterior_mean_candidate_points,
                    posterior_std_candidate_points,
                    y_star,
                    float("inf"),
                )
            elif self.task == "max":
                truncated_gaussian = tfp.distributions.TruncatedNormal(
                    posterior_mean_candidate_points,
                    posterior_std_candidate_points,
                    float("-inf"),
                    y_star,
                )
            self.D_acquisition[es][1] = truncated_gaussian.mean() / equal_cost(es)
        return self.D_acquisition

    def _suspected_intervention_this_trial(self) -> tuple[list[str], tf.Tensor]:
        # Find the es and corresponding candidate points with the highest acquisition function value (expected improvement)
        # return the suspected optimal intervention
        max_ei = float("-inf")
        for es in self.exploration_set:
            candidate_points, ei = self.D_acquisition[es]
            ei_max_this_es = tf.reduce_max(ei)
            ei_max_index = tf.argmax(ei, axis=0)
            candidate_points_max = candidate_points[ei_max_index]
            if ei_max_this_es > max_ei:
                max_ei = ei_max_this_es
                suspected_es = es
                suspected_candidate_point = candidate_points_max
        return suspected_es, suspected_candidate_point

    def _intervene_and_augment(
        self,
        temporal_index: int,
        suspected_es: list[str],
        suspected_candidate_point: tf.Tensor,
    ) -> None:
        intervention = {
            key: [None] * (temporal_index + 1) for key in self.sem.static().keys()
        }
        if temporal_index >= 1:
            for t in range(temporal_index - 1):
                es, decision_values = self.D_intervene_history[t]["decision_vars"]
                for node, value in zip(es, decision_values):
                    intervention[node][t] = value

        for node in suspected_es:
            intervention[node][temporal_index] = suspected_candidate_point

        samples = draw_samples_from_sem(
            self.sem,
            self.num_monte_carlo,
            temporal_index + 1,
            intervention=intervention,
        )
        mean_samples = OrderedDict()
        for key, value in samples.items():
            mean_samples[key] = tf.reduce_mean(value, axis=0)
        # Update the intervention history
        if self.D_interven[temporal_index][suspected_es] is None:
            self.D_interven[temporal_index][suspected_es] = mean_samples
        else:
            for key, value in mean_samples.items():
                self.D_interven[temporal_index][suspected_es][key] = tf.concat(
                    [self.D_interven[temporal_index][suspected_es][key], value], axis=0
                )

    def _update_sem_hat(self, temporal_index: int) -> None:
        self.dynamic_graph.temporal_index = temporal_index
        fcns_full = fcns4sem(self.dynamic_graph, self.D_obs)
        self.sem_estimated = sem_hat(fcns_full)()

    def _update_observational_data(self, temporal_index: int) -> None:
        # currently, the full observational data is available at each time step
        # before the construction of the DynCausalBayesOpt object.
        # For the future, more should be done to update the observational data.
        # For now, we just pass
        pass
