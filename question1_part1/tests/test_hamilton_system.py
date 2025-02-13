import unittest
from prob_models import IndepedentGaussians, OneDimGaussianMixtureDensity
from hamilton_system import HamiltonianSystem
import tensorflow as tf

class TestHamiltonianSystem(unittest.TestCase):

    def test_H_and_derivatives(self):
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        H_system = HamiltonianSystem(expU, expK)
        q = tf.constant([-0.3])
        p = tf.constant([0.7])
        H = H_system.H(q, p)
        dHdp = H_system.dHdp(q, p)
        dHdq = H_system.dHdq(q, p)
        assert H.shape == () and dHdp.shape == (1, ) and dHdq.shape == (1, )
        q = tf.ones((3, 1))
        p = tf.ones((3, 1))
        H = H_system.H(q, p)
        dHdp = H_system.dHdp(q, p)
        dHdq = H_system.dHdq(q, p)
        assert H.shape == (3, ) and dHdp.shape == (3, 1) and dHdq.shape == (3, 1)
