import unittest
from pdf_models import IndepedentGaussians, OneDimGaussianMixtureDensity, ThreeDimRosenbrock
from hamilton_system import HamiltonianSystem
import tensorflow as tf

class TestHamiltonianSystem(unittest.TestCase):

    def test_H_and_derivatives_1d(self):
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        H_system = HamiltonianSystem(expU, expK)
        q = tf.constant([-0.3])
        p = tf.constant([0.7])
        H = H_system.H(q, p)
        dHdp = H_system.dHdp(q, p)
        dHdq = H_system.dHdq(q, p)
        self.assertEqual(H.shape, (1,))
        self.assertEqual(dHdp.shape, (1,))
        self.assertEqual(dHdq.shape, (1,))
        q = tf.ones((3, 1))
        p = tf.ones((3, 1))
        H = H_system.H(q, p)
        dHdp = H_system.dHdp(q, p)
        dHdq = H_system.dHdq(q, p)
        self.assertEqual(H.shape, (3, ))
        self.assertEqual(dHdp.shape, (3, 1))
        self.assertEqual(dHdq.shape, (3, 1))

    def test_symplectic_integrate_1d(self):
        expU = OneDimGaussianMixtureDensity()
        expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
        H_system = HamiltonianSystem(expU, expK)
        hist = H_system.symplectic_integrate(tf.constant([1.0]), tf.constant([3.0]), 0.05, 400)
        self.assertEqual(hist.shape, (401, 4))
        self.assertAlmostEqual(hist[0, 0].numpy(), 1.0)
        self.assertAlmostEqual(hist[0, 1].numpy(), 3.0)

    def test_H_and_derivatives_3d(self):
        expU = ThreeDimRosenbrock()
        expK = IndepedentGaussians(tf.constant([0.0, 0.0, 0.0]), tf.constant([1.0, 1.0, 1.0]))
        H_system = HamiltonianSystem(expU, expK)
        q = tf.constant([-0.3, 0.2, 0.1])
        p = tf.constant([0.7, -0.5, 0.3])
        H = H_system.H(q, p)
        dHdp = H_system.dHdp(q, p)
        dHdq = H_system.dHdq(q, p)
        self.assertEqual(H.shape, (1,))
        self.assertEqual(dHdp.shape, (3,))
        self.assertEqual(dHdq.shape, (3,))
        q = tf.ones((5, 3))
        p = tf.ones((5, 3))
        H = H_system.H(q, p)
        dHdp = H_system.dHdp(q, p)
        dHdq = H_system.dHdq(q, p)
        self.assertEqual(H.shape, (5, ))
        self.assertEqual(dHdp.shape, (5, 3))
        self.assertEqual(dHdq.shape, (5, 3))


    def test_symplectic_integrate_3d(self):
        expU = ThreeDimRosenbrock()
        expK = IndepedentGaussians(tf.constant([0.0, 0.0, 0.0]), tf.constant([1.0, 1.0, 1.0]))
        H_system = HamiltonianSystem(expU, expK)
        hist = H_system.symplectic_integrate(tf.constant([1.0, 1.0, 1.0]), tf.constant([3.0, 3.0, 3.0]), 0.05, 50)
        self.assertEqual(hist.shape, (51, 12))
        self.assertAlmostEqual(hist[0, 0].numpy(), 1.0)
        self.assertAlmostEqual(hist[0, 4].numpy(), 3.0)