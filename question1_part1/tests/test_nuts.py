import unittest
import tensorflow as tf
from no_u_turn.nuts import NoUTurnSampling
from hamilton_neural_network import LatentHamiltonianNeuralNetwork
import sys
sys.path.append("..")


class TestNoUTurnSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting NoUTurnSampling tests...")

    @classmethod
    def tearDownClass(cls):
        print("Finished NoUTurnSampling tests...")

    def test_leapfrog(self):
        lhnn = LatentHamiltonianNeuralNetwork(2, 16, 4, None, None)
        lhnn.build(input_shape=(1, 2))
        lhnn.summary()
        lhnn.load_weights("./exps/demo_1_lhnn.weights.h5")
        num_samples = 100
        q0 = tf.constant([[0.0]])
        p0 = tf.constant([[0.0]])
        dt = 0.05
        nuts = NoUTurnSampling(num_samples=num_samples, q0=q0, dt=dt, lhnn=lhnn)
        q_prime, p_prime = nuts.leapfrog(q0, p0, 1)
        self.assertEqual(q_prime.shape, q0.shape)
        self.assertEqual(p_prime.shape, p0.shape)
