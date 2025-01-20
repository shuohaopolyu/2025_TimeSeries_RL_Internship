from utils import build_mix_gaussian_variable, build_mix_gaussian_function
import tensorflow_probability as tfp
import tensorflow as tf
import unittest
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

class TestMixGaussian(unittest.TestCase):
    def test_build_mix_gaussian_variable(self):
        y = tfd.Chi(df=3.0).sample((10000,))
        num_mix = 30
        learning_rate = 0.001
        max_training_step = 10000
        final_mixture = build_mix_gaussian_variable(
            y=y,
            num_mix=num_mix,
            learning_rate=learning_rate,
            max_training_step=max_training_step,
            debug_mode=True,
        )
        self.assertIsInstance(final_mixture, tfd.MixtureSameFamily)
        # fig, axs = plt.subplots(1,2)
        # sns.histplot(y, ax=axs[0], kde=True)
        # sns.histplot(final_mixture.sample(10000), ax=axs[1], kde=True)
        # plt.show()
        self.assertEqual(final_mixture.sample(100).shape, (100,))

    def test_build_mix_gaussian_function(self):
        x = tf.linspace(-4.0, 4.0, 1000)
        y = 2.0 * tf.tanh(x) + tfd.Chi(df=3.0).sample((1000,)) * 0.8

        num_mix = 100
        learning_rate = 0.001
        max_training_step = 20000

        conditional_gmm = build_mix_gaussian_function(
            x=x,
            y=y,
            num_mix=num_mix,
            learning_rate=learning_rate,
            max_training_step=max_training_step,
        )

        pred_y_cd_x = conditional_gmm.sample(x, num_samples=10)
        # fig, axs = plt.subplots(1,2)
        # axs[0].plot(x, y)
        # axs[1].plot(x, tf.transpose(pred_y_cd_x), alpha=0.3, color="red")
        # plt.show()
        self.assertEqual(pred_y_cd_x.shape, (10, 1000))


