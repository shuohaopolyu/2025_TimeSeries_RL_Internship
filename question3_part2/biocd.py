from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp
from utils import build_gprm, build_mix_gaussian_variable, build_mix_gaussian_function
import matplotlib.pyplot as plt
import seaborn as sns
import time

tfd = tfp.distributions


class BIOCausalDiscovery:
    """Bayesian Intervention Optimization for Causal Discovery"""

    def __init__(
        self,
        true_sem: callable,
        D_obs: OrderedDict,
        D_int: OrderedDict = OrderedDict(),
        num_int: int = 10,
        k_0: float = 10.0,
        k_1: float = 0.1,
        beta: float = 0.2,
        num_monte_carlo: int = 4096,
        num_mixture: int = 50,
        num_samples_per_sem: int = 100000,
        intervention_domain: list = None,
        max_iter: int = 20000,
        learning_rate: float = 0.001,
        patience: int = 20,
        debug_mode: bool = False,
        time_eval: bool = False,
    ):
        self.true_sem = true_sem
        self.D_obs = D_obs
        self.num_int = num_int
        self.D_int = D_int
        self.k_0 = k_0
        self.k_1 = k_1
        self.beta = beta
        self.num_monte_carlo = num_monte_carlo
        self.num_mix = num_mixture
        self.num_samples_per_sem = num_samples_per_sem
        self.intervention_domain = (
            [tf.reduce_min(D_obs["X"]), tf.reduce_max(D_obs["X"])]
            if intervention_domain is None
            else intervention_domain
        )
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.patience = patience
        self.debug_mode = debug_mode
        self.time_eval = time_eval

    def run(self):
        if not hasattr(self, "m_0"):
            self._update_m_0()
        if not hasattr(self, "m_1"):
            self._update_m_1()
        for i in range(self.num_int):
            self._update_bayes_factor_01_int()
            self._update_prior_p_dc()
            if self.time_eval:
                time_start = time.time()
                self._find_x_opt()
                print(f"Time for finding x_opt: {time.time() - time_start}")
            else:
                self._find_x_opt()
            print(
                f"Optimal intervention: {'%.2f' % self.x_opt.numpy()}, Max p_dc: {'%.2f' % self.max_p_dc.numpy()}, ",
                f"LogBF01: {'%.2f' % tf.math.log(self.bayes_factor_01_int).numpy()}, ",
                f"Prior of H0: {'%.2f' % self.p_h0_D_int}, Prior of H1: {'%.2f' % self.p_h1_D_int}",
            )
            self._recorder()
            self._update_D_int()

    def _p_dc(self, x: tf.Tensor) -> tf.Tensor:
        y_0 = self.m_0.sample((self.num_monte_carlo,))
        y_1 = self.m_1.sample(x, self.num_monte_carlo)[:, 0]
        _x = x * tf.ones((self.num_monte_carlo,))
        bf_01_y_0 = self._bayes_factor_01(y_0, _x)
        bf_01_y_1 = self._bayes_factor_01(y_1, _x)

        p_dc = (
            tf.reduce_mean(tf.exp(-1.0 / self.beta * tf.nn.relu(self.k_0 - bf_01_y_0)))
            * self.p_h0_D_int
            + tf.reduce_mean(
                tf.exp(-1.0 / self.beta * tf.nn.relu(bf_01_y_1 - self.k_1))
            )
            * self.p_h1_D_int
        )
        return p_dc

    def _find_x_opt(self):
        x = tfd.Uniform(
            self.intervention_domain[0], self.intervention_domain[1]
        ).sample((100,))
        results = tf.map_fn(self._p_dc, x)
        self.max_p_dc = tf.reduce_max(results)
        self.x_opt = x[tf.argmax(results)]
        return self.x_opt, self.max_p_dc
    
    def _ini_max_p_dc(self):
        x = tfd.Uniform(
            self.intervention_domain[0], self.intervention_domain[1]
        ).sample((100,))
        self._update_m_0()
        self._update_m_1()
        self._update_bayes_factor_01_int()
        self._update_prior_p_dc()
        results = tf.map_fn(self._p_dc, x)
        return tf.reduce_max(results)

    def _recorder(self):
        if not hasattr(self, "max_p_dc_record"):
            self.max_p_dc_record = []
            self.bayes_factor_01_int_record = []
            self.p_h0_D_int_record = []
            self.p_h1_D_int_record = []
        self.max_p_dc_record.append(self.max_p_dc.numpy())
        self.bayes_factor_01_int_record.append(
            tf.math.log(self.bayes_factor_01_int).numpy()
        )
        self.p_h0_D_int_record.append(self.p_h0_D_int)
        self.p_h1_D_int_record.append(self.p_h1_D_int)

    def recorder_plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        axs[0, 0].plot(self.max_p_dc_record)
        axs[0, 0].set_title("Max p_dc")
        axs[0, 1].plot(self.bayes_factor_01_int_record)
        axs[0, 1].set_title("log Bayes Factor")
        axs[1, 0].plot(self.p_h0_D_int_record)
        axs[1, 0].set_title("Prior of H0 condition on D_int")
        axs[1, 1].plot(self.p_h1_D_int_record)
        axs[1, 1].set_title("Prior of H1 condition on D_int")
        plt.show()

    def get_recorder_data(self):
        result = OrderedDict()
        result["max_p_dc"] = self.max_p_dc_record
        result["bayes_factor_01_int"] = self.bayes_factor_01_int_record
        result["p_h0_D_int"] = self.p_h0_D_int_record
        result["p_h1_D_int"] = self.p_h1_D_int_record
        return result

    def _bayes_factor_01(self, y: tf.Tensor, x: tf.Tensor):
        m_0_y = self.m_0.prob(y)
        m_1_yx = self.m_1.prob(y, x)
        return (m_0_y / m_1_yx) * self.bayes_factor_01_int

    def _update_D_int(self):
        x_opt = self.x_opt
        if not self.D_int:
            self.D_int["X"] = tf.reshape(x_opt, (1,))
            self.D_int["Y"] = tf.reduce_mean(
                self.true_sem.propagate(self.num_samples_per_sem, x_opt)["Y"],
                keepdims=True,
            )
        else:
            new_x = tf.reshape(x_opt, (1,))
            self.D_int["X"] = tf.concat([self.D_int["X"], new_x], axis=0)
            new_y = tf.reduce_mean(
                self.true_sem.propagate(self.num_samples_per_sem, x_opt)["Y"],
                keepdims=True,
            )
            self.D_int["Y"] = tf.concat([self.D_int["Y"], new_y], axis=0)

    def _update_bayes_factor_01_int(self):
        if not self.D_int:
            self.bayes_factor_01_int = 1.0
            return self.bayes_factor_01_int
        else:
            m_0_y = self.m_0.prob(self.D_int["Y"])
            m_1_yx = self.m_1.prob(self.D_int["Y"], self.D_int["X"])
            self.bayes_factor_01_int = tf.reduce_prod(m_0_y / m_1_yx)
            return self.bayes_factor_01_int

    def _update_m_0(self):
        self.m_0 = build_mix_gaussian_variable(
            y=self.D_obs["Y"],
            num_mix=self.num_mix,
            learning_rate=self.learning_rate,
            max_training_step=self.max_iter,
            debug_mode=self.debug_mode,
        )
        # sns.kdeplot(self.m_0.sample((5000,)))
        # sns.kdeplot(self.D_obs["Y"])
        # plt.show()
        return self.m_0

    def _update_m_1(self):
        self.m_1 = build_mix_gaussian_function(
            x=self.D_obs["X"],
            y=self.D_obs["Y"],
            num_mix=self.num_mix,
            learning_rate=self.learning_rate,
            max_training_step=self.max_iter,
            debug_mode=self.debug_mode,
        )
        return self.m_1

    def _update_prior_p_dc(self, p_h0=0.5, p_h1=0.5):
        if not self.D_int:
            self.p_h0_D_int = 0.5
            self.p_h1_D_int = 0.5
            return 0.5, 0.5
        else:
            p_D_int_h0 = tf.reduce_prod(self.m_0.prob(self.D_int["Y"]))
            p_D_int_h1 = tf.reduce_prod(self.m_1.prob(self.D_int["Y"], self.D_int["X"]))
            self.p_h0_D_int = (
                p_D_int_h0 * p_h0 / (p_D_int_h0 * p_h0 + p_D_int_h1 * p_h1)
            )
            self.p_h1_D_int = 1.0 - self.p_h0_D_int
            return self.p_h0_D_int, self.p_h1_D_int

    def _update_m_1_dev(self):
        # Update the model m_1 using Gaussian Process Regression,
        # but currently, this function is replaced by mixture Gaussian function for
        # the purpose of reproducing the results in the paper.
        index_x = (
            tf.linspace(
                tf.reduce_min(self.D_obs["X"]), tf.reduce_max(self.D_obs["X"]), 100
            )
        )[:, tf.newaxis]
        m_1 = build_gprm(
            index_x,
            self.D_obs["X"][:, tf.newaxis],
            self.D_obs["Y"],
            amplitude_factor=1.0,
            length_scale_factor=10.0,
            obs_noise_factor=2.0,
            max_training_step=self.max_iter,
            learning_rate=self.learning_rate,
            patience=self.patience,
            debug_mode=self.debug_mode,
        )
        return m_1
