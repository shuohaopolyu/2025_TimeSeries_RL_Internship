import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from utils.sequential_sampling import (
    draw_samples_from_sem_dev,
    draw_samples_from_sem_hat_dev,
)
from utils.costs import equal_cost
from utils.sem_estimate import sem_hat, fy_and_fny, label_pairs, fcns4sem
from utils.gaussian_process import build_gprm
import matplotlib.pyplot as plt

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
        self.D_interven = OrderedDict((t, OrderedDict()) for t in range(self.T))
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
        for es in self.exploration_set:
            self.causal_gpm_list[tuple(es)] = None
        self.D_acquisition = OrderedDict()

    def run(self):
        """Run Dynamic Causal Bayesian Optimization"""
        opt_history = [[] for _ in range(self.T)]
        for temporal_index in range(self.T):
            # Update the observational data
            self._update_observational_data(temporal_index)

            # Update the estimated SEM model
            self._update_sem_hat(temporal_index)

            # initialise dynamic causal GP models
            self._prior_causal_gp(temporal_index)

            print(
                "Dynamic causal Bayesian optimization at time step {} is started.".format(
                    temporal_index
                )
            )

            # Initialize the exploration set
            for trial in range(self.num_trials):
                self.trial_index = trial
                # initialise the posterior causal GP models
                self._posterior_causal_gp(temporal_index)
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
                # Update the optimal intervention history
                self._update_opt_intervene_history(temporal_index)

                opt_history[temporal_index].append(
                    self.opt_intervene_history[temporal_index]["optimal_value"]
                )



                print("Temporal index:", temporal_index, ". Trial:", trial)
                print(
                    "Intervened exploration set:",
                    suspected_es,
                    ". Intervention point:",
                    suspected_candidate_point.numpy(),
                )
                print(
                    "Target variable value:",
                    self.D_interven[temporal_index][suspected_es][self.target_var][
                        -1, temporal_index
                    ].numpy(),
                )
                print(
                    "Optimal value:",
                    self.opt_intervene_history[temporal_index]["optimal_value"],
                )

            print(
                "Dynamic causal Bayesian optimization at time step {} is completed.".format(
                    temporal_index
                )
            )
        return opt_history
    
    def _intervention_points(self, es) -> tf.Tensor:
        # Generate the candidate points for each exploration set within the intervention domain
        # return the candidate points
        if len(es) == 1:
            intervention_min, intervention_max = self.intervention_domain[es[0]]
            candidate_points = tf.sort(tf.random.uniform(
                (self.num_anchor_points, 1), intervention_min, intervention_max
            ), axis=0)
        else:
            for i, node in enumerate(es):
                if i == 0:
                    intervention_min, intervention_max = self.intervention_domain[node]
                    candidate_points = tf.random.uniform(
                        (self.num_anchor_points, 1), intervention_min, intervention_max
                    )
                else:
                    intervention_min, intervention_max = self.intervention_domain[node]
                    candidate_points = tf.concat(
                        [
                            candidate_points,
                            tf.random.uniform(
                                (self.num_anchor_points, 1),
                                intervention_min,
                                intervention_max,
                            ),
                        ],
                        axis=1,
                    )
        return candidate_points

    def _initialize_exploration_set(self) -> list[tuple[str]]:
        self.dyn_graph.temporal_index = 0
        mis = self.dyn_graph.minimal_intervene_set()
        # filter out the empty set
        exploration_set = [es for es in mis if es]
        # Create a new exploration set with modified node identifiers
        general_exploration_set = []
        for subset in exploration_set:
            # Create a new subset by taking the part before '_' in each node identifier
            new_subset = [node.split("_")[0] for node in subset]
            general_exploration_set.append(tuple(new_subset))
        return general_exploration_set

    def _optimal_intervene_value(
        self, temporal_index: int
    ) -> tuple[tuple[str], list, float]:
        i_D_interven = self.D_interven[temporal_index]
        # print(i_D_interven)
        if len(i_D_interven) == 0:
            if self.task == "min":
                return None, None, float("inf")
            elif self.task == "max":
                return None, None, float("-inf")
        extreme_values = []
        for _, sub_dict in i_D_interven.items():
            if sub_dict is None:
                continue
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

    def _update_opt_intervene_vars(self, temporal_index: int) -> None:
        opt_es = self.opt_intervene_history[temporal_index]["decision_vars"][0]
        for node in opt_es:
            self.full_opt_intervene_vars.append(node + "_{}".format(temporal_index))

    def _update_opt_intervene_history(self, temporal_index: int) -> None:
        es, es_values, global_extreme = self._optimal_intervene_value(temporal_index)
        self.opt_intervene_history[temporal_index] = {
            "decision_vars": (es, es_values),
            "optimal_value": global_extreme,
        }

    def _input_fny(
        self, the_graph: object, samples: tf.Tensor, temporal_index: int
    ) -> tf.Tensor:
        # temporal_index = samples[self.target_var].shape[1] - 1
        current_node = self.target_var + "_" + str(temporal_index)
        predecessors = list(the_graph.predecessors(current_node))
        ny_nodes = []
        for predecessor in predecessors:
            predecessor_index = int(predecessor.split("_")[1])
            if predecessor_index == temporal_index:
                ny_nodes.append(predecessor)
        return label_pairs(samples, current_node, ny_nodes)[0]

    def _intervene_scheme(
        self,
        x_py: list[str],
        i_py: list[str],
        x_py_values: tf.Tensor,
        temporal_index: int,
    ) -> dict:

        max_time_step = temporal_index + 1
        sem_keys = list(self.sem.static().keys())
        intervention_scheme = []
        for i in range(x_py_values.shape[0]):
            intervention = {key: [None] * max_time_step for key in sem_keys}
            for node in i_py:
                key = node.split("_")[0]
                node_index = int(node.split("_")[1])
                opt_es, opt_es_values = self.opt_intervene_history[node_index][
                    "decision_vars"
                ]
                intervention[key][node_index] = opt_es_values[opt_es.index(key)]
            for j, node in enumerate(x_py):
                key = node.split("_")[0]
                node_index = int(node.split("_")[1])
                intervention[key][node_index] = x_py_values[i, j]
            if temporal_index > 0:
                for t in range(temporal_index):
                    opt_es, opt_es_values = self.opt_intervene_history[t][
                        "decision_vars"
                    ]
                    for ii, node in enumerate(opt_es):
                        intervention[node][t] = opt_es_values[ii]
            intervention_scheme.append(intervention)
        return intervention_scheme

    def _mean_fny_xiw(
        self,
        x_py_values: tf.Tensor,
        x_py: list[str],
        i_py: list[str],
        temporal_index: int,
        fny_fcn: list[callable],
    ) -> tf.Tensor:
        # x_py_values is a tensor of shape (e_num, len(x_py))
        # will return a tensor of shape (e_num, num_monte_carlo)
        intervention = self._intervene_scheme(x_py, i_py, x_py_values, temporal_index)
        for i, i_intervention in enumerate(intervention):
            i_samples = draw_samples_from_sem_hat_dev(
                self.sem_estimated,
                self.num_monte_carlo,
                temporal_index,
                intervention=i_intervention,
                seed=None,
            )
            self.dyn_graph.temporal_index = temporal_index
            the_graph = self.dyn_graph.graph.copy()
            i_input_fny = self._input_fny(the_graph, i_samples, temporal_index)
            # print("i_samples", i_samples)
            i_samples_mean_fny_xiw = fny_fcn[0](i_input_fny)[tf.newaxis, :]
            i_samples_std_fny_xiw = fny_fcn[1](i_input_fny)[tf.newaxis, :]
            # print("i_samples_mean_fny_xiw", i_samples_mean_fny_xiw)
            # print("i_samples_std_fny_xiw", i_samples_std_fny_xiw)
            i_samples_fny_xiw = tfp.distributions.Normal(
                i_samples_mean_fny_xiw, i_samples_std_fny_xiw
            ).sample()
            if i == 0:
                # samples_mean_fny_xiw = i_samples_mean_fny_xiw
                samples_mean_fny_xiw = i_samples_fny_xiw
            else:
                # print(samples_mean_fny_xiw.shape, i_samples_mean_fny_xiw.shape)
                samples_mean_fny_xiw = tf.concat(
                    [samples_mean_fny_xiw, i_samples_fny_xiw], axis=0
                )
        return samples_mean_fny_xiw

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
                f_star = tf.constant(
                    [[self.opt_intervene_history[temporal_index - 1]["optimal_value"]]]
                )
                # compute the mean and standard deviation of fy(f_star)
                mean_fy_f_star = fy_fcn[0](f_star)
                std_fy_f_star = fy_fcn[1](f_star)
                # print("mean_fy_f_star", mean_fy_f_star.shape)
                # print("std_fy_f_star", std_fy_f_star.shape)
                # exclude the y_pt from the predecessor
                predecessor.remove(self.target_var + "_" + str(temporal_index - 1))
                samples_fy_f_star = tfp.distributions.Normal(
                    mean_fy_f_star[0], std_fy_f_star[0]
                ).sample((1, self.num_monte_carlo))
            else:
                samples_fy_f_star = None

            x_py = [(x + "_" + str(temporal_index)) for x in es]
            # get the subset of the predecessor that subscript is less than temporal_index
            previous_py = [
                node for node in predecessor if int(node.split("_")[1]) < temporal_index
            ]
            i_py = [
                node
                for node in previous_py
                if node.split("_")[0] in self.self.full_opt_intervene_vars
            ]

            def prior_mean(
                x_py_values: tf.Tensor, x_py=x_py, i_py=i_py, fny_fcn=fny_fcn
            ):
                # print("x_py_values_ini", x_py_values.shape)
                samples_mean_fny_xiw = self._mean_fny_xiw(
                    x_py_values, x_py, i_py, temporal_index, fny_fcn
                )
                e_num = samples_mean_fny_xiw.shape[0]
                if fy_fcn[0] is not None:
                    samples_fy_f_star_tile = tf.tile(
                        samples_fy_f_star, [e_num, 1]
                    )
                    # print("samples_fy_f_star_tile", samples_fy_f_star_tile)
                    # print("samples_mean_fny_xiw", samples_mean_fny_xiw)
                    mean_val = tf.reduce_mean(
                        samples_mean_fny_xiw + samples_fy_f_star_tile, axis=[1]
                    )
                    return mean_val
                else:
                    mean_val = tf.reduce_mean(samples_mean_fny_xiw, axis=[1])
                    return mean_val

            def prior_std(
                x_py_values: tf.Tensor, x_py=x_py, i_py=i_py, fny_fcn=fny_fcn
            ):
                samples_mean_fny_xiw = self._mean_fny_xiw(
                    x_py_values, x_py, i_py, temporal_index, fny_fcn
                )
                e_num = samples_mean_fny_xiw.shape[0]
                if fy_fcn[0] is not None:
                    samples_fy_f_star_tile = tf.tile(
                        samples_fy_f_star, [e_num, 1]
                    )
                    std_val = tf.math.reduce_std(
                        samples_mean_fny_xiw + samples_fy_f_star_tile, axis=[1]
                    )
                    return std_val
                else:
                    std_val = tf.math.reduce_std(samples_mean_fny_xiw, axis=[1])
                    return std_val

            self.prior_causal_gp[es] = (prior_mean, prior_std)
        return self.prior_causal_gp

    def _posterior_causal_gp(self, temporal_index: int) -> OrderedDict:
        self.candidate_points_dict = OrderedDict()
        for es in self.exploration_set:
            self.candidate_points_dict[es] = self._intervention_points(es)
            if es not in self.D_interven[temporal_index]:
                # No intervention is performed
                def prior_mean(index_x=self.candidate_points_dict[es]):
                    return self.prior_causal_gp[es][0](index_x)
                def prior_std(index_x=self.candidate_points_dict[es]):
                    return self.prior_causal_gp[es][1](index_x)
                self.posterior_causal_gp[es] = (prior_mean, prior_std)
            else:
                # Intervention is performed
                es_interven = self.D_interven[temporal_index][es]
                es_intervene_x = []
                for i, key in enumerate(es):
                    if i == 0:
                        es_intervene_x = (es_interven[key][:, temporal_index])[
                            :, tf.newaxis
                        ]
                    else:
                        es_intervene_x = tf.concat(
                            (
                                es_intervene_x,
                                (es_interven[key][:, temporal_index])[:, tf.newaxis],
                            ),
                            axis=1,
                        )
                es_intervene_y = es_interven[self.target_var][:, temporal_index]
                causal_gpm, _, _ = build_gprm(
                    index_x=self.candidate_points_dict[es],
                    x=es_intervene_x,
                    y=es_intervene_y,
                    mean_fn=self.prior_causal_gp[es][0],
                    causal_std_fn=self.prior_causal_gp[es][1],
                    obs_noise_factor=1e-2,
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
                    debug_mode=False,
                    learning_rate=1e-4,
                )
                
                def posterior_mean(causal_gpm=causal_gpm):
                    return causal_gpm.mean()

                def posterior_std(causal_gpm=causal_gpm):
                    return causal_gpm.stddev()

                self.posterior_causal_gp[es] = (posterior_mean, posterior_std)

        return self.posterior_causal_gp


    def _acquisition_function(self, temporal_index: int) -> OrderedDict:
        self.D_acquisition = OrderedDict()
        # this figure is used for "stationary" case study
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        for es_idx, es in enumerate(self.exploration_set):
            self.D_acquisition[es] = [None, None]
            candidate_points = self.candidate_points_dict[es]
            self.D_acquisition[es][0] = candidate_points
            _, _, y_star = self._optimal_intervene_value(temporal_index)
            posterior_mean_candidate_points = (
                self.posterior_causal_gp[es][0]()
            )
            posterior_std_candidate_points = (
                self.posterior_causal_gp[es][1]()
            )
            y_star = y_star * tf.ones_like(posterior_mean_candidate_points)

            axs[es_idx].plot(candidate_points, posterior_mean_candidate_points[:, None])
            axs[es_idx].fill_between(
                candidate_points[:, 0],
                posterior_mean_candidate_points
                - 1.96 * posterior_std_candidate_points,
                posterior_mean_candidate_points
                + 1.96 * posterior_std_candidate_points,
                alpha=0.2,
            )
            if temporal_index in self.D_interven:
                if es in self.D_interven[temporal_index]:
                    axs[es_idx].scatter(
                        self.D_interven[temporal_index][es][es[0]][:-1, temporal_index],
                        self.D_interven[temporal_index][es][self.target_var][:-1, temporal_index],
                        color="red",
                        marker="x",
                    )

            axs[es_idx].set_title("Posterior of the causal GP for exploration set {}".format(es))
            if temporal_index == 0:
                axs[es_idx].set_ylim(-5,5)
            elif temporal_index == 1:
                axs[es_idx].set_ylim(-7,4)
            elif temporal_index == 2:
                axs[es_idx].set_ylim(-10,4)

            axs[es_idx].set_xlim(
                self.intervention_domain[es[0]][0], self.intervention_domain[es[0]][1]
            )

            axs[es_idx].set_xlabel(es[0])
            axs[es_idx].set_ylabel(self.target_var)

            # Expected improvement
            if self.task == "min":
                z = (
                    y_star - posterior_mean_candidate_points
                ) / posterior_std_candidate_points
                # print("z", z)
                self.D_acquisition[es][1] = (
                    y_star - posterior_mean_candidate_points
                ) * tfp.distributions.Normal(0.0, 1.0).cdf(
                    z
                ) + posterior_std_candidate_points * tfp.distributions.Normal(
                    0.0, 1.0
                ).prob(
                    z
                )

            elif self.task == "max":
                z = (
                    posterior_mean_candidate_points - y_star
                ) / posterior_std_candidate_points
                self.D_acquisition[es][1] = (
                    posterior_mean_candidate_points - y_star
                ) * tfp.distributions.Normal(0.0, 1.0).cdf(
                    z
                ) + posterior_std_candidate_points * tfp.distributions.Normal(
                    0.0, 1.0
                ).prob(
                    z
                )

        plt.savefig("./experiments/stat_{}_{}.png".format(temporal_index, self.trial_index), dpi=300, bbox_inches="tight")
        plt.close()

            # print("acquisition function value", self.D_acquisition[es][1])
        return self.D_acquisition

    def _suspected_intervention_this_trial(self) -> tuple[list[str], tf.Tensor]:
        # Find the es and corresponding candidate points with the highest acquisition function value (expected improvement)
        # return the suspected optimal intervention
        _ei = float("-inf")
        for es in self.exploration_set:
            candidate_points, ei = self.D_acquisition[es]
            ei_max_this_es = tf.reduce_max(ei, axis=0)
            ei_max_index = tf.argmax(ei, axis=0)
            candidate_points_max = candidate_points[ei_max_index, :]
            if ei_max_this_es > _ei:
                _ei = ei_max_this_es
                suspected_es = es
                suspected_candidate_point = candidate_points_max
        return suspected_es, suspected_candidate_point[tf.newaxis, :]

    def _intervene_and_augment(
        self,
        temporal_index: int,
        suspected_es: tuple[str],
        suspected_candidate_point: tf.Tensor,
    ) -> None:
        intervention = {
            key: [None] * (temporal_index + 1) for key in self.sem.static().keys()
        }

        for t in range(temporal_index):
            es, decision_values = self.opt_intervene_history[t]["decision_vars"]
            for node, value in zip(es, decision_values):
                intervention[node][t] = tf.constant(value, dtype=tf.float32)

        for i, node in enumerate(suspected_es):
            intervention[node][temporal_index] = suspected_candidate_point[0, i]

        samples = draw_samples_from_sem_dev(
            self.sem,
            num_samples=1,
            temporal_index=temporal_index,
            intervention=intervention,
            epsilon=0.0,
        )

        mean_samples = OrderedDict()
        for key, value in samples.items():
            mean_samples[key] = tf.reduce_mean(value, axis=0, keepdims=True)
        # Update the intervention history
        if temporal_index not in self.D_interven:
            self.D_interven[temporal_index] = OrderedDict()
        if suspected_es not in self.D_interven[temporal_index]:
            self.D_interven[temporal_index][suspected_es] = mean_samples
        else:
            for key, value in mean_samples.items():
                self.D_interven[temporal_index][suspected_es][key] = tf.concat(
                    [self.D_interven[temporal_index][suspected_es][key], value], axis=0
                )

    def _update_sem_hat(self, temporal_index: int) -> None:
        self.dyn_graph.temporal_index = temporal_index
        fcns_full = fcns4sem(self.dyn_graph.graph, self.D_obs)
        self.sem_estimated = sem_hat(fcns_full)()

    def _update_observational_data(self, temporal_index: int) -> None:
        # currently, the full observational data is available at each time step
        # before the construction of the DynCausalBayesOpt object.
        # For the future, more should be done to update the observational data.
        # For now, we just pass
        pass
