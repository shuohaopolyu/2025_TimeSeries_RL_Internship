import unittest
from utils.gaussian_process import (
    build_gprm,
    build_gaussian_process,
    build_gaussian_variable,
)
from utils.causal_kernel import CausalKernel
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
        data_x = tf.random.normal([100, 1])
        sample = OrderedDict([("X", data_x), ("Z", [])])
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


class TestCausalKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for utils.CausalKernel")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for utils.CausalKernel")

    def test_CausalKernel_returns(self):
        def tested_causal_std_fn(x):
            return tf.math.reduce_sum(x, axis=-1)

        kernel = CausalKernel(tested_causal_std_fn, 1.0, 1.0)
        x1 = tf.random.normal([10, 1])
        apply_result_1 = kernel.apply(x1, x1)
        self.assertIsInstance(apply_result_1, tf.Tensor)
        self.assertEqual(apply_result_1.shape, (10, 10))

    def test_test_build_causal_gaussian_process(self):

        def tested_causal_std_fn(x):
            return tf.squeeze(tf.math.reduce_sum(x, axis=-1))

        def tested_mean_fn(x):
            return tf.squeeze(tf.math.reduce_sum(x, axis=-1))

        x = tf.linspace(-1.0, 1.0, 20)
        y = tf.sin(x * 3.14) + tf.random.normal([20], 0, 0.1)
        x = x[..., tf.newaxis]
        index_x = tf.constant([[0.0], [1.0]], dtype=tf.float32)
        causalgpm, _, _ = build_gprm(
            index_x=index_x,
            x=x,
            y=y,
            mean_fn=tested_mean_fn,
            causal_std_fn=tested_causal_std_fn,
            amplitude_factor=1.0,
            length_scale_factor=1.0,
            obs_noise_factor=1.0,
            max_training_step=20000,
            learning_rate=2e-4,
            patience=20,
            debug_mode=True,
        )

        self.assertIsInstance(
            causalgpm, tfp.distributions.GaussianProcessRegressionModel
        )
        self.assertIsInstance(causalgpm.mean(), tf.Tensor)
