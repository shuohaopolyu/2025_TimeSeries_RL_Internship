import unittest
import tensorflow as tf
from no_u_turn.nuts import NoUTurnSampling
from hamilton_neural_network import LatentHamiltonianNeuralNetwork
import matplotlib.pyplot as plt


class TestNoUTurnSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting NoUTurnSampling tests...")
        cls.lhnn = LatentHamiltonianNeuralNetwork(3, 16, 4)
        cls.lhnn.build((1, 2))
        cls.lhnn.load_weights("./exps/demo_1_lhnn.weights.h5")

    @classmethod
    def tearDownClass(cls):
        print("Finished NoUTurnSampling tests...")

    def test_leapfrog(self):
        num_samples = 300
        q0 = tf.constant([[0.0]])
        p0 = tf.constant([[1.0]])
        dt = 0.05
        nuts = NoUTurnSampling(num_samples=num_samples, q0=q0, dt=dt, lhnn=self.lhnn)
        q_prime, p_prime = nuts.leapfrog(q0, p0, 1)
        self.assertEqual(q_prime.shape, q0.shape)
        self.assertEqual(p_prime.shape, p0.shape)
        hist = []
        for i in range(num_samples):
            hist.append(q_prime)
            q_prime, p_prime = nuts.leapfrog(q_prime, p_prime, 1)
        hist = tf.concat(hist, axis=0)
        plt.plot(hist.numpy().flatten())
        plt.show()


    # def test_call(self):
    #     q0 = tf.constant([[1.0]])
    #     nuts = NoUTurnSampling(num_samples=200, q0=q0, dt=0.05, lhnn=self.lhnn)
    #     nuts()
    #     # self.assertEqual(len(nuts.q_hist), 101)
    #     q_hist = tf.concat(nuts.q_hist, axis=0)
    #     plt.hist(q_hist.numpy().flatten(), bins=30)
    #     plt.show()

    # def test_buildtree(self):
    #     q0 = tf.constant([[0.0]])
    #     p0 = tf.constant([[1.0]])
    #     H = self.lhnn.forward(q0, p0)
    #     u = tf.random.uniform([], 0, tf.exp(-H))
    #     nuts = NoUTurnSampling(num_samples=200, q0=q0, dt=0.05, lhnn=self.lhnn)
    #     q_minus, p_minus, q_plus, p_plus, C_prime, s_prime = (
    #         nuts.buildtree(q0, p0, u, 1, 0)
    #     )
    #     C = tf.concat([c[0] for c in C_prime], axis=0)
    #     print(len(C))
    #     plt.plot(C.numpy().flatten())
    #     plt.show()
    #     print(f"q_minus: {q_minus}")
    #     print(f"p_minus: {p_minus}")
