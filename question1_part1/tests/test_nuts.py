import unittest
import tensorflow as tf
from no_u_turn.nuts import NoUTurnSampling
from hamilton_neural_network import LatentHamiltonianNeuralNetwork
from pdf_models import IndepedentGaussians, OneDimGaussianMixtureDensity
from hamilton_system import HamiltonianSystem


class TestNoUTurnSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting NoUTurnSampling tests...")
        cls.lhnn = LatentHamiltonianNeuralNetwork(3, 16, 2)
        cls.lhnn.build((1, 2))
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        cls.Hamiltonian = HamiltonianSystem(expU, expK)
        num_samples = 300
        q0 = tf.constant([[0.0]])
        dt = 0.05
        cls.nuts = NoUTurnSampling(
            num_samples=num_samples,
            q0=q0,
            dt=dt,
            lhnn=cls.lhnn,
            Hamiltonian=cls.Hamiltonian,
            Delta_lf=1000.0,
            Delta_lhnn=10.0,
            num_lf_steps=20,
        )

    @classmethod
    def tearDownClass(cls):
        print("Finished NoUTurnSampling tests...")

    def test_hnn_leapfrog(self):
        q0 = tf.constant([[0.0]])
        p0 = tf.constant([[1.0]])
        q_prime, p_prime = self.nuts.leapfrog(q0, p0, 1)
        self.assertEqual(q_prime.shape, q0.shape)
        self.assertEqual(p_prime.shape, p0.shape)

    def test_leapfrog(self):
        q0 = tf.constant([[0.0]])
        p0 = tf.constant([[1.0]])
        q_prime, p_prime = self.nuts.leapfrog(q0, p0, -1)
        self.assertEqual(q_prime.shape, q0.shape)
        self.assertEqual(p_prime.shape, p0.shape)
        
    # def test_call(self):
    #     q0 = tf.constant([[0.0]])
    #     nuts = NoUTurnSampling(num_samples=500, q0=q0, dt=0.05, lhnn=self.lhnn)
    #     nuts()
    #     # q_hist = tf.concat(nuts.q_hist, axis=0)
    #     # plt.hist(q_hist.numpy()[1000:, 0].flatten(), bins=30)
    #     # plt.show()
    #     self.assertEqual(len(nuts.q_hist), nuts.num_samples + 1)

    # def test_buildtree(self):
    #     q0 = tf.constant([[0.0]])
    #     p0 = tf.constant([[1.0]])
    #     H = self.lhnn.forward(q0, p0)
    #     u = tf.random.uniform([], 0, tf.exp(-H))
    #     nuts = NoUTurnSampling(num_samples=200, q0=q0, dt=0.01, lhnn=self.lhnn)
    #     q_minus, p_minus, q_plus, p_plus, C_prime, s_prime = nuts.buildtree(
    #         q0, p0, u, 1, 5
    #     )
    #     C = tf.concat([c[0] for c in C_prime], axis=0)
    #     symplectic = (self.lhnn.symplectic_integrate(q0, p0, 0.01, 32))[1:, :1]
    #     # plt.plot(symplectic.numpy())
    #     # plt.plot(C.numpy())
    #     # plt.show()
    #     # compare the leapfrog integrator with the symplectic integrator
    #     self.assertTrue(
    #         tf.reduce_mean(tf.abs(symplectic - C)) < 1e-4,
    #         "The leapfrog integrator is not accurate.",
    #     )
    #     self.assertEqual(q_minus.shape, q0.shape)
    #     self.assertEqual(p_minus.shape, p0.shape)
    #     self.assertEqual(q_plus.shape, q0.shape)
    #     self.assertEqual(p_plus.shape, p0.shape)
