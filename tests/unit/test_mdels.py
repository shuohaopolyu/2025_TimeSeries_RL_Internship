import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import unittest
from equations.stationary import StationaryModel
from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class TestStationaryModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for moodels.stat.StationaryModel")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for moodels.stat.StationaryModel")

    def test_static(self):
        static_model = StationaryModel.static()
        # compute vertice X
        noise = tfd.Normal(loc=0.0, scale=1.0).sample()
        X = static_model["X"](noise, t=0, sample=OrderedDict())
        self.assertIsInstance(X, tf.Tensor)
        self.assertEqual(X, noise)
        # compute vertice Z
        noise = tfd.Normal(loc=0.0, scale=1.0).sample()
        sample = OrderedDict([("X", tf.constant([1.0, 2.0, 3.0]))])
        Z = static_model["Z"](noise, t=1, sample=sample)
        true_Z = tf.exp(-sample["X"][1]) + noise
        self.assertIsInstance(Z, tf.Tensor)
        self.assertEqual(Z, true_Z)
        # compute vertice Y
        noise = tfd.Normal(loc=0.0, scale=1.0).sample()
        sample = OrderedDict([("Z", tf.constant([1.0, 2.0, 3.0]))])
        Y = static_model["Y"](noise, t=2, sample=sample)
        true_Y = tf.cos(sample["Z"][2]) - tf.exp(-sample["Z"][2] / 20.0) + noise
        self.assertIsInstance(Y, tf.Tensor)
        self.assertEqual(Y, true_Y)

    def test_dynamic(self):
        model = StationaryModel.dynamic()
        # compute vertice X
        noise = tfd.Normal(loc=0.0, scale=1.0).sample()
        sample = OrderedDict([("X", tf.constant([1.0]))])
        X = model["X"](noise, t=1, sample=sample)
        true_X = sample["X"][0] + noise
        self.assertIsInstance(X, tf.Tensor)
        self.assertEqual(X, true_X)
        # compute vertice Z
        noise = tfd.Normal(loc=0.0, scale=1.0).sample()
        sample = OrderedDict([("X", tf.constant([1.0, 2.0])), ("Z", tf.constant([1.0]))])
        Z = model["Z"](noise, t=1, sample=sample)
        true_Z = tf.exp(-sample["X"][1]) + sample["Z"][0] + noise
        self.assertIsInstance(Z, tf.Tensor)
        self.assertEqual(Z, true_Z)
        # compute vertice Y
        noise = tfd.Normal(loc=0.0, scale=1.0).sample()
        sample = OrderedDict([("Z", tf.constant([1.0, 2.0])), ("Y", tf.constant([1.0]))])
        Y = model["Y"](noise, t=1, sample=sample)
        true_Y = tf.cos(sample["Z"][1]) - tf.exp(-sample["Z"][1] / 20.0) + sample["Y"][0] + noise
        self.assertIsInstance(Y, tf.Tensor)
        self.assertEqual(Y, true_Y)