import unittest
import causal_graph.utils as utils
from utils.gaussian_process import build_gprm
import tensorflow as tf
import tensorflow_probability as tfp


class TestBuild_gprm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = tf.linspace(-1.0, 1.0, 100)
        cls.y = tf.sin(x * 3.14) + tf.random.normal([100], 0, 0.1)
        cls.x = tf.reshape(x, [-1, 1])
        index_x = tf.linspace(-1.0, 1.0, 100)
        cls.index_x = tf.reshape(index_x, [-1, 1])
        print("Starting tests for utils.gaussian_process")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for utils.gaussian_process")

    def test_build_gprm_returns(self):
        gprm, _, _ = build_gprm(self.index_x, self.x, self.y)
        self.assertIsInstance(gprm, tfp.distributions.GaussianProcessRegressionModel)

    def test_para_opt(self):
        gprm, _, _ = build_gprm(self.index_x, self.x, self.y)
        noise_tol = 3e-3
        self.assertAlmostEqual(
            gprm.observation_noise_variance, 0.01, delta=noise_tol
        )
