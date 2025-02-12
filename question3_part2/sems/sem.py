import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict

tfd = tfp.distributions


class X2Y:
    def __init__(
        self,
        ini_mean_dist=tfd.Uniform(-4.0, 4.0),
        ini_var_dist=tfd.Chi2(3.0),
        total_k_num=10,
        A=2.0,
        B=1.0,
    ):
        self._ini_mean_dist = ini_mean_dist
        self._ini_var_dist = ini_var_dist
        self._total_k_num = total_k_num
        self._nx_dist = self._ini_random_dist()
        self._ny_dist = self._ini_random_dist()
        self._A = A
        self._B = B

    def _ini_random_dist(self):
        std_normal = tfd.Normal(0.0, 1.0)
        z = std_normal.sample((self._total_k_num,))
        pi = tf.nn.softmax(z) * 0.5 + 1 / (2 * self._total_k_num)
        mean = self._ini_mean_dist.sample((self._total_k_num,))
        var = self._ini_var_dist.sample((self._total_k_num,))
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi),
            components_distribution=tfd.Normal(loc=mean, scale=tf.sqrt(var)),
        )

    def propagate(self, n_samples, x_intervention: float = None):
        if x_intervention is None:
            X = self._nx_dist.sample((n_samples,))
            Y = self._A * tf.tanh(X * self._B) + self._ny_dist.sample((n_samples,))
        else:
            X = x_intervention * tf.ones((n_samples,))
            Y = self._A * tf.tanh(X * self._B) + self._ny_dist.sample((n_samples,))
        return OrderedDict((("X", X), ("Y", Y)))


class Y2X:
    def __init__(
        self,
        ini_mean_dist=tfd.Uniform(-4.0, 4.0),
        ini_var_dist=tfd.Chi2(3.0),
        total_k_num=10,
        A=2.0,
        B=1.0,
    ):
        self._ini_mean_dist = ini_mean_dist
        self._ini_var_dist = ini_var_dist
        self._total_k_num = total_k_num
        self._ny_dist = self._ini_random_dist()
        self._nx_dist = self._ini_random_dist()
        self._A = A
        self._B = B

    def _ini_random_dist(self):
        std_normal = tfd.Normal(0.0, 1.0)
        z = std_normal.sample((self._total_k_num,))
        pi = tf.nn.softmax(z) * 0.5 + 1 / (2 * self._total_k_num)
        mean = self._ini_mean_dist.sample((self._total_k_num,))
        var = self._ini_var_dist.sample((self._total_k_num,))
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi),
            components_distribution=tfd.Normal(loc=mean, scale=tf.sqrt(var)),
        )

    def propagate(self, n_samples, x_intervention: float = None):
        if x_intervention is None:
            Y = self._ny_dist.sample((n_samples,))
            X = self._A * tf.tanh(Y * self._B) + self._nx_dist.sample((n_samples,))
        else:
            Y = self._ny_dist.sample((n_samples,))
            X = x_intervention * tf.ones((n_samples,))
        return OrderedDict((("Y", Y), ("X", X)))


class U2XY:
    def __init__(
        self,
        ini_mean_dist=tfd.Uniform(-4.0, 4.0),
        ini_var_dist=tfd.Chi2(3.0),
        total_k_num=10,
        A=2.0,
        B=1.0,
        C=2.0,
        D=1.0,
    ):
        self._ini_mean_dist = ini_mean_dist
        self._ini_var_dist = ini_var_dist
        self._total_k_num = total_k_num
        self._nu_dist = self._ini_random_dist()
        self._nx_dist = self._ini_random_dist()
        self._ny_dist = self._ini_random_dist()
        self._A = A
        self._B = B
        self._C = C
        self._D = D

    def _ini_random_dist(self):
        std_normal = tfd.Normal(0.0, 1.0)
        z = std_normal.sample((self._total_k_num,))
        pi = tf.nn.softmax(z) * 0.5 + 1 / (2 * self._total_k_num)
        mean = self._ini_mean_dist.sample((self._total_k_num,))
        var = self._ini_var_dist.sample((self._total_k_num,))
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi),
            components_distribution=tfd.Normal(loc=mean, scale=tf.sqrt(var)),
        )

    def propagate(self, n_samples, x_intervention: float = None):
        if x_intervention is None:
            U = self._nu_dist.sample((n_samples,))
            Y = self._A * tf.tanh(U * self._B) + self._ny_dist.sample((n_samples,))
            X = self._C * tf.tanh(U * self._D) + self._nx_dist.sample((n_samples,))
        else:
            U = self._nu_dist.sample((n_samples,))
            Y = self._A * tf.tanh(U * self._B) + self._ny_dist.sample((n_samples,))
            X = x_intervention * tf.ones((n_samples,))
        return OrderedDict((("U", U), ("Y", Y), ("X", X)))
