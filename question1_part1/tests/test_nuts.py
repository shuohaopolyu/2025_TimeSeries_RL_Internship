import unittest
import tensorflow as tf
from no_u_turn.nuts import NoUTurnSampling
from hamilton_neural_network import LatentHamiltonianNeuralNetwork
import matplotlib.pyplot as plt

class TestNoUTurnSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting NoUTurnSampling tests...")
        cls.lhnn = LatentHamiltonianNeuralNetwork(3, 16, 4, None, None)
        cls.lhnn.load_weights("./exps/demo_1_lhnn.weights.h5")

    @classmethod
    def tearDownClass(cls):
        print("Finished NoUTurnSampling tests...")

    def test_leapfrog(self):
        num_samples = 100
        q0 = tf.constant([[0.0]])
        p0 = tf.constant([[0.0]])
        dt = 0.05
        nuts = NoUTurnSampling(num_samples=num_samples, q0=q0, dt=dt, lhnn=self.lhnn)
        q_prime, p_prime = nuts.leapfrog(q0, p0, 1)
        self.assertEqual(q_prime.shape, q0.shape)
        self.assertEqual(p_prime.shape, p0.shape)

    def test_call(self):
        q0 = tf.constant([[1.0]])
        nuts = NoUTurnSampling(num_samples=200, q0=q0, dt=0.05, lhnn=self.lhnn)
        nuts()
        # self.assertEqual(len(nuts.q_hist), 101)
        q_hist = tf.concat(nuts.q_hist, axis=0)
        plt.hist(q_hist.numpy().flatten(), bins=30)
        plt.show()

