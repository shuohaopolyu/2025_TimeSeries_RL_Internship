from hamilton_neural_network import (
    TrainTestData,
    HamiltonianNeuralNetwork,
    LatentHamiltonianNeuralNetwork,
)
from pdf_models import IndepedentGaussians, OneDimGaussianMixtureDensity
import unittest
import tensorflow as tf


class TestTrainTestData(unittest.TestCase):

    def test_TrainTestData(self):
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        q0 = tf.constant([0.0])
        p0 = tf.constant([1.0])
        T = 20.0
        leap_frog_per_unit = 20
        num_samples = 40
        train_test_data = TrainTestData(
            num_samples, expU, expK, T, leap_frog_per_unit, q0, p0
        )
        samples = train_test_data()
        self.assertEqual(samples.shape, (num_samples, 4))

class TestHamiltonianNeuralNetwork(unittest.TestCase):
    
    def test_call(self):
        hnn_0 = HamiltonianNeuralNetwork(2, 32, None, None)
        q = tf.constant([[1.0]])
        p = tf.constant([[1.0]])
        h = hnn_0(q, p)
        self.assertEqual(h.shape, (1, 1))

        hnn_1 = HamiltonianNeuralNetwork(2, 32, None, None)
        q = tf.constant([[1.0, 2.0]])
        p = tf.constant([[1.0, 2.0]])
        h = hnn_1(q, p)
        self.assertEqual(h.shape, (1, 1))

        hnn_2 = HamiltonianNeuralNetwork(2, 32, None, None)
        q = tf.constant([[1.0, 1.2], [1.0, 1.2]])
        p = tf.constant([[1.0, 1.2], [1.0, 1.2]])
        h = hnn_2(q, p)
        self.assertEqual(h.shape, (2, 1))

    def test_loss(self):
        hnn_0 = HamiltonianNeuralNetwork(2, 32, None, None)
        q = tf.constant([[1.0]])
        p = tf.constant([[1.0]])
        dqdt = tf.constant([[1.0]])
        dpdt = tf.constant([[1.0]])
        loss = hnn_0.loss(q, p, dqdt, dpdt)
        self.assertEqual(loss.shape, (1, ))

class TestLatentHamiltonianNeuralNetwork(unittest.TestCase):
    pass