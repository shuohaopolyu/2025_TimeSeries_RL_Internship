import unittest
from prob_models import IndepedentGaussians, OneDimGaussianMixtureDensity
import tensorflow as tf

class TestIndepedentGaussians(unittest.TestCase):

    def test_single_input(self):
        mus = tf.constant([1.0, 2.0, 3.0])
        sigmas = tf.constant([1.0, 1.0, 1.0])
        model = IndepedentGaussians(mus, sigmas)
        q = tf.constant([1.0, 2.0, 3.0])
        pdf = model.f(q)
        self.assertEqual(pdf.shape, ())
        self.assertAlmostEqual(pdf.numpy(), 1.0 / (2 * 3.14159) ** 1.5, places=4)

    def test_batch_input(self):
        mus = tf.constant([1.0, 2.0, 3.0])
        sigmas = tf.constant([1.0, 1.0, 1.0])
        model = IndepedentGaussians(mus, sigmas)
        q1 = tf.ones((10,1))
        q2 = 2.0*tf.ones((10,1))
        q3 = 3.0*tf.ones((10,1))
        q = tf.concat([q1, q2, q3], axis=1)
        pdf = model.f(q)
        self.assertEqual(pdf.shape, (10,))
        self.assertAlmostEqual(pdf.numpy().sum(), 10.0 / (2 * 3.14159) ** 1.5, places=4)

    def test_input_wrong_shape(self):
        mus = tf.constant([1.0, 2.0, 3.0])
        sigmas = tf.constant([1.0, 1.0, 1.0])
        model = IndepedentGaussians(mus, sigmas)
        q = tf.ones((10, 2))
        with self.assertRaises(AssertionError):
            pdf = model.f(q)

class TestOneDimGaussianMixtureDensity(unittest.TestCase):

    def test_single_input(self):
        model = OneDimGaussianMixtureDensity(sigma_1=0.35, sigma_2=0.35, mu_1=1, mu_2=1)
        q = tf.constant([1.0])
        pdf = model.f(q)
        self.assertEqual(pdf.shape, ())
        self.assertAlmostEqual(pdf.numpy(), 1.0)

    def test_batch_input(self):
        model = OneDimGaussianMixtureDensity()
        q = tf.ones((10,1))
        pdf = model.f(q)
        self.assertEqual(pdf.shape, (10,))

    def test_input_wrong_shape(self):
        model = OneDimGaussianMixtureDensity()
        q = tf.ones((10, 2))
        with self.assertRaises(AssertionError):
            pdf = model.f(q)
