from utils import ConditionalMixGaussian
import tensorflow_probability as tfp
import tensorflow as tf
import unittest
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

class TestConditionalDist(unittest.TestCase):
    def test_p_y_given_x(self):
        psi = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.7, 0.3]),
            components_distribution=tfd.Normal(loc=[0.1, -0.1], scale=[0.9, 1.4])
        )
        theta = (2.0, 1.0)
        model = ConditionalMixGaussian(psi, theta)

        p_y_cd_x = model.p_y_given_x(tf.constant([1.0]))
        self.assertIsInstance(p_y_cd_x, tfd.MixtureSameFamily)
        self.assertEqual(p_y_cd_x.batch_shape, [1])
        self.assertEqual(p_y_cd_x.sample(100).shape, (100, 1))

        p_y_cd_x_2 = model.p_y_given_x(tf.constant([1.0, 2.0]))
        self.assertIsInstance(p_y_cd_x_2, tfd.MixtureSameFamily)
        self.assertEqual(p_y_cd_x_2.batch_shape, [2])
        self.assertEqual(p_y_cd_x_2.sample(100).shape, (100, 2))

    def test_log_prob(self):
        psi = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.7, 0.3]),
            components_distribution=tfd.Normal(loc=[0.1, -0.1], scale=[0.9, 1.4])
        )
        theta = (2.0, 1.0)
        model = ConditionalMixGaussian(psi, theta)

        log_prob = model.log_prob(tf.constant([1.0, 2.0]), tf.constant([1.0, 2.0]))
        self.assertEqual(log_prob.shape, (2,))

    def test_prob(self):
        psi = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.7, 0.3]),
            components_distribution=tfd.Normal(loc=[0.001, -0.001], scale=[1.0, 1.0])
        )
        theta = (0.0, 1.0)
        model = ConditionalMixGaussian(psi, theta)

        prob = model.prob(tf.constant([1.0, 2.0]), tf.constant([1.0, 2.0]))
        self.assertEqual(prob.shape, (2,))
        self.assertTrue(prob[0] > prob[1])

        psi = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
            components_distribution=tfd.Normal(loc=[0.0, 0.0], scale=[1.0, 1.0])
        )
        theta = (1.0, 1.0)
        model = ConditionalMixGaussian(psi, theta)

        prob = model.prob(tf.constant([1.0, 2.0]), tf.constant([1.0, 2.0]))
        residual = tf.stack([1.0 - tf.tanh(1.0), 2.0 - tf.tanh(2.0)], axis=0)
        truth = tfd.Normal(loc=0.0, scale=1.0).prob(residual)
        self.assertTrue(tf.reduce_all(tf.abs(prob - truth) < 1e-6))

    def test_sample(self):
        psi = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.7, 0.3]),
            components_distribution=tfd.Normal(loc=[0.1, -0.1], scale=[0.9, 1.4])
        )
        theta = (2.0, 1.0)
        model = ConditionalMixGaussian(psi, theta)

        sample = model.sample(tf.constant([1.0, 2.0]), num_samples=100)
        self.assertEqual(sample.shape, (100, 2))
