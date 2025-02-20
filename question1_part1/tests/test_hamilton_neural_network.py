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
        num_samples = 2
        train_test_data = TrainTestData(
            num_samples, expU, expK, T, leap_frog_per_unit, q0, p0
        )
        samples = train_test_data()
        self.assertEqual(
            samples.shape, (int(num_samples * (T * leap_frog_per_unit + 1)), 4)
        )


class TestHamiltonianNeuralNetwork(unittest.TestCase):

    def test_call(self):
        hnn_0 = HamiltonianNeuralNetwork(2, 32, None, None)
        hnn_0.build(input_shape=(1, 2))
        q = tf.constant([[1.0]])
        p = tf.constant([[1.0]])
        inputs = tf.concat([q, p], axis=-1)
        h = hnn_0(inputs)
        self.assertEqual(h.shape, (1, 1))

        hnn_1 = HamiltonianNeuralNetwork(2, 32, None, None)
        hnn_1.build(input_shape=(1, 4))
        q = tf.constant([[1.0, 2.0]])
        p = tf.constant([[1.0, 2.0]])
        h = hnn_1.forward(q, p)
        self.assertEqual(h.shape, (1, 1))

        hnn_2 = HamiltonianNeuralNetwork(2, 32, None, None)
        hnn_2.build(input_shape=(1, 4))
        q = tf.constant([[1.0, 1.2], [1.0, 1.2]])
        p = tf.constant([[1.0, 1.2], [1.0, 1.2]])
        h = hnn_2.forward(q, p)
        self.assertEqual(h.shape, (2, 1))

    def test_loss(self):
        hnn_0 = HamiltonianNeuralNetwork(2, 32, None, None)
        q = tf.constant([[1.0]])
        p = tf.constant([[1.0]])
        dqdt = tf.constant([[1.0]])
        dpdt = tf.constant([[1.0]])
        loss = hnn_0.loss_fcn(q, p, dqdt, dpdt)
        self.assertEqual(loss.shape, ())

    def test_train(self):
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        q0 = tf.constant([0.0])
        p0 = tf.constant([1.0])
        T = 2.0
        leap_frog_per_unit = 20
        num_samples = 2
        train_test_data = TrainTestData(
            num_samples, expU, expK, T, leap_frog_per_unit, q0, p0
        )
        samples = train_test_data()
        train_set = samples[:60, :]
        test_set = samples[60:, :]
        hnn = HamiltonianNeuralNetwork(2, 32, train_set, test_set)
        hnn.build(input_shape=(1, 2))
        hnn.summary()
        hnn.train(epochs=200, batch_size=10)


class TestLatentHamiltonianNeuralNetwork(unittest.TestCase):

    def test_call(self):
        lhnn_0 = LatentHamiltonianNeuralNetwork(2, 32, 4, None, None)
        lhnn_0.build(input_shape=(1, 2))
        q = tf.constant([[1.0]])
        p = tf.constant([[1.0]])
        inputs = tf.concat([q, p], axis=-1)
        z = lhnn_0(inputs)
        self.assertEqual(z.shape, (1, 4))

        lhnn_1 = LatentHamiltonianNeuralNetwork(2, 32, 4, None, None)
        lhnn_1.build(input_shape=(1, 4))
        q = tf.constant([[1.0, 2.0]])
        p = tf.constant([[1.0, 2.0]])
        inputs = tf.concat([q, p], axis=-1)
        z = lhnn_1(inputs)
        self.assertEqual(z.shape, (1, 4))

        lhnn_2 = LatentHamiltonianNeuralNetwork(2, 32, 4, None, None)
        lhnn_2.build(input_shape=(1, 4))
        q = tf.constant([[1.0, 1.2], [1.0, 1.2]])
        p = tf.constant([[1.0, 1.2], [1.0, 1.2]])
        inputs = tf.concat([q, p], axis=-1)
        z = lhnn_2(inputs)
        self.assertEqual(z.shape, (2, 4))

    def test_loss(self):
        lhnn_0 = LatentHamiltonianNeuralNetwork(2, 32, 4, None, None)
        lhnn_0.build(input_shape=(1, 4))
        q = tf.constant([[1.0, 2.0]])
        p = tf.constant([[1.0, 2.0]])
        dqdt = tf.constant([[1.0, 2.0]])
        dpdt = tf.constant([[1.0, 2.0]])
        loss = lhnn_0.loss_fcn(q, p, dqdt, dpdt)
        self.assertEqual(loss.shape, ())

    def test_train(self):
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        q0 = tf.constant([0.0])
        p0 = tf.constant([1.0])
        T = 1.0
        leap_frog_per_unit = 20
        num_samples = 4
        train_test_data = TrainTestData(
            num_samples, expU, expK, T, leap_frog_per_unit, q0, p0
        )
        samples = train_test_data()
        train_set = samples[:30, :]
        test_set = samples[30:, :]
        lhnn = LatentHamiltonianNeuralNetwork(2, 32, 4, train_set, test_set)
        lhnn.build(input_shape=(1, 2))
        lhnn.summary()
        lhnn.train(epochs=300, batch_size=10)
