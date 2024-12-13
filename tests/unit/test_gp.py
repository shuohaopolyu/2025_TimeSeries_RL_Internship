import unittest
from utils.gaussian_process import build_gprm, build_gaussian_process, build_gaussian_variable
import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict


class TestBuild_gprm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for utils.gaussian_process")
        x = tf.linspace(-1.0, 1.0, 100)
        cls.y = tf.sin(x * 3.14) + tf.random.normal([100], 0, 0.1)
        cls.x = tf.reshape(x, [-1, 1])
        index_x = tf.linspace(-1.0, 1.0, 100)
        cls.index_x = tf.reshape(index_x, [-1, 1])

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for utils.gaussian_process")

    def test_build_gprm_returns(self):
        gprm, _, _ = build_gprm(self.index_x, self.x, self.y)
        self.assertIsInstance(gprm, tfp.distributions.GaussianProcessRegressionModel)
        # noise_tol = 3e-3
        # self.assertAlmostEqual(gprm.observation_noise_variance, 0.01, delta=noise_tol)

    def test_build_gaussian_process(self):
        gprm, _, _ = build_gprm(self.index_x, self.x, self.y)
        predecessors = ["X_0"]
        sample = OrderedDict([("X", [0.5, 0.6]), ("Z", [])])
        gp_fcn = build_gaussian_process(gprm, predecessors)
        self.assertIsInstance(gp_fcn(sample), tf.Tensor)


class TestBuild_gaussian_variable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for utils.build_gaussian_variable")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for utils.build_gaussian_variable")

    def test_build_gprm_returns(self):
        obs_data = tf.random.normal([100], 0, 0.1)
        gaussian_variable = build_gaussian_variable(obs_data)
        self.assertIsInstance(gaussian_variable(0), tf.Tensor)

