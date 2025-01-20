from sems import Y2X, U2XY, X2Y
import tensorflow_probability as tfp
import tensorflow as tf
import unittest
from collections import OrderedDict
tfd = tfp.distributions

class TestY2X(unittest.TestCase):
    def test_ini_random_dist(self):
        y2x = Y2X()
        self.assertIsInstance(y2x._ny_dist, tfd.MixtureSameFamily)
        self.assertEqual(y2x._ny_dist.sample((10,)).shape, (10,))

    def test_propagate(self):
        y2x = Y2X()
        res = y2x.propagate(5000)
        self.assertIsInstance(res, OrderedDict)
        self.assertIn('X', res)
        self.assertIn('Y', res)
        self.assertEqual(res['X'].shape, (5000,))
        self.assertEqual(res['Y'].shape, (5000,))
        self.assertFalse(tf.reduce_all(tf.equal(res['X'], res['Y'])))

class TestU2XY(unittest.TestCase):
    def test_ini_random_dist(self):
        u2xy = U2XY()
        self.assertIsInstance(u2xy._nu_dist, tfd.MixtureSameFamily)
        self.assertEqual(u2xy._nu_dist.sample((10,)).shape, (10,))

    def test_propagate(self):
        u2xy = U2XY()
        res = u2xy.propagate(5000)
        self.assertIsInstance(res, OrderedDict)
        self.assertIn('X', res)
        self.assertIn('Y', res)
        self.assertEqual(res['X'].shape, (5000,))
        self.assertEqual(res['U'].shape, (5000,))
        self.assertFalse(tf.reduce_all(tf.equal(res['X'], res['Y'])))
        self.assertFalse(tf.reduce_all(tf.equal(res['X'], res['U'])))

class TestX2Y(unittest.TestCase):
    def test_ini_random_dist(self):
        x2y = X2Y()
        self.assertIsInstance(x2y._nx_dist, tfd.MixtureSameFamily)
        self.assertEqual(x2y._nx_dist.sample((10,)).shape, (10,))

    def test_propagate(self):
        x2y = X2Y()
        res = x2y.propagate(5000)
        self.assertIsInstance(res, OrderedDict)
        self.assertIn('X', res)
        self.assertIn('Y', res)
        self.assertEqual(res['X'].shape, (5000,))
        self.assertEqual(res['Y'].shape, (5000,))
        self.assertFalse(tf.reduce_all(tf.equal(res['X'], res['Y'])))
