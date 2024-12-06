from utils.sequential_sampling import sample_from_sem, draw_samples_from_sem
import unittest
import tensorflow as tf
from collections import OrderedDict
from sem.stationary import StationaryModel


class TestSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for utils.sequential_sampling")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for utils.sequential_sampling")

    def round_tensor(self, tensor, decimals):
        return tf.round(tensor * 10**decimals) / 10**decimals

    def test_sample_from_sem(self):
        sem = StationaryModel()
        # test 1: no intervention
        epsilon = OrderedDict(
            [
                ("X", tf.constant([0.1, 0.1, 0.1])),
                ("Z", tf.constant([0.1, 0.1, 0.1])),
                ("Y", tf.constant([0.1, 0.1, 0.1])),
            ]
        )
        sample = sample_from_sem(sem, 3, epsilon=epsilon)
        self.assertEqual(len(sample), 3)
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    self.round_tensor(sample["X"], 2), tf.constant([0.1, 0.2, 0.3])
                )
            )
        )
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    self.round_tensor(sample["Z"], 2), tf.constant([1.00, 1.92, 2.76])
                )
            )
        )
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    self.round_tensor(sample["Y"], 2),
                    tf.constant([-0.31, -1.47, -3.17]),
                )
            )
        )

        # test 2: with intervention
        intervention = {
            "X": [0.5, None, None],
            "Z": [None, 0.5, None],
            "Y": [None, None, None],
        }
        epsilon = OrderedDict(
            [
                ("X", tf.constant([0.1, 0.1, 0.1])),
                ("Z", tf.constant([0.1, 0.1, 0.1])),
                ("Y", tf.constant([0.1, 0.1, 0.1])),
            ]
        )
        sample = sample_from_sem(sem, 3, intervention=intervention, epsilon=epsilon)
        self.assertEqual(len(sample), 3)
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    self.round_tensor(sample["X"], 2), tf.constant([0.5, 0.6, 0.7])
                )
            )
        )
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    self.round_tensor(sample["Z"], 2), tf.constant([0.71, 0.5, 1.10])
                )
            )
        )
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    self.round_tensor(sample["Y"], 2),
                    tf.constant([-0.10, -0.10, -0.49]),
                )
            )
        )

    def test_draw_samples_from_sem(self):
        # test 1: no intervention
        sem = StationaryModel()
        samples = draw_samples_from_sem(sem, 3, 3)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples["X"].shape, (3, 3))
        self.assertEqual(samples["Z"].shape, (3, 3))
        self.assertEqual(samples["Y"].shape, (3, 3))
        self.assertIsInstance(samples["X"], tf.Tensor)
        self.assertIsInstance(samples["Z"], tf.Tensor)
        self.assertIsInstance(samples["Y"], tf.Tensor)

        # test 2: with intervention
        intervention = {
            "X": [0.5, None, None],
            "Z": [None, 0.5, None],
            "Y": [None, None, None],
        }
        samples = draw_samples_from_sem(sem, 3, 3, intervention=intervention)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples["X"].shape, (3, 3))
        self.assertEqual(samples["Z"].shape, (3, 3))
        self.assertEqual(samples["Y"].shape, (3, 3))
        self.assertIsInstance(samples["X"], tf.Tensor)
        self.assertIsInstance(samples["Z"], tf.Tensor)
        self.assertIsInstance(samples["Y"], tf.Tensor)