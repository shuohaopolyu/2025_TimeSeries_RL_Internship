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
        self.assertEqual(loss.shape, ())

    def test_train(self):
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        q0 = tf.constant([0.0])
        p0 = tf.constant([1.0])
        T = 2.0
        leap_frog_per_unit = 20
        num_samples = 40
        train_test_data = TrainTestData(
            num_samples, expU, expK, T, leap_frog_per_unit, q0, p0
        )
        samples = train_test_data()
        train_set = samples[:30, :]
        test_set = samples[30:, :]
        hnn = HamiltonianNeuralNetwork(2, 32, train_set, test_set)
        hnn(tf.constant([[1.0]]), tf.constant([[1.0]]))
        hnn.summary()
        hnn.train(epochs=1000, batch_size=10)

class TestLatentHamiltonianNeuralNetwork(unittest.TestCase):

    def test_call(self):
        lhnn_0 = LatentHamiltonianNeuralNetwork(2, 32, 4, None, None)
        q = tf.constant([[1.0]])
        p = tf.constant([[1.0]])
        z = lhnn_0(q, p)
        self.assertEqual(z.shape, (1, 4))

        lhnn_1 = LatentHamiltonianNeuralNetwork(2, 32, 4, None, None)
        q = tf.constant([[1.0, 2.0]])
        p = tf.constant([[1.0, 2.0]])
        z = lhnn_1(q, p)
        self.assertEqual(z.shape, (1, 4))

        lhnn_2 = LatentHamiltonianNeuralNetwork(2, 32, 4, None, None)
        q = tf.constant([[1.0, 1.2], [1.0, 1.2]])
        p = tf.constant([[1.0, 1.2], [1.0, 1.2]])
        z = lhnn_2(q, p)
        self.assertEqual(z.shape, (2, 4))

    def test_loss(self):
        lhnn_0 = LatentHamiltonianNeuralNetwork(2, 32, 4, None, None)
        q = tf.constant([[1.0, 2.0]])
        p = tf.constant([[1.0, 2.0]])
        dqdt = tf.constant([[1.0, 2.0]])
        dpdt = tf.constant([[1.0, 2.0]])
        loss = lhnn_0.loss(q, p, dqdt, dpdt)
        self.assertEqual(loss.shape, ())

    def test_train(self):
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        q0 = tf.constant([0.0])
        p0 = tf.constant([1.0])
        T = 2.0
        leap_frog_per_unit = 20
        num_samples = 40
        train_test_data = TrainTestData(
            num_samples, expU, expK, T, leap_frog_per_unit, q0, p0
        )
        samples = train_test_data()
        train_set = samples[:30, :]
        test_set = samples[30:, :]
        lhnn = LatentHamiltonianNeuralNetwork(2, 32, 4, train_set, test_set)
        lhnn(tf.constant([[1.0]]), tf.constant([[1.0]]))
        lhnn.summary()
        lhnn.train(epochs=1000, batch_size=10)
