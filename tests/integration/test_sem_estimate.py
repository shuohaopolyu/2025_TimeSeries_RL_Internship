import unittest
import tensorflow as tf
import tensorflow_probability as tfp
from utils.sem_estimate import fcns4sem, sem_hat, fy_and_fny
from causal_graph.dynamic_graph import DynCausalGraph
from causal_graph.example_dyn_graphs import three_step_stat

from utils.sequential_sampling import (
    sample_from_sem,
    draw_samples_from_sem,
    sample_from_sem_hat,
    draw_samples_from_sem_hat,
)
from sem.stationary import StationaryModel
from collections import OrderedDict

seed = 1111
tf.random.set_seed(seed)


class TestSemEstimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for utils.sem_estimate")
        cls.tested_graph = three_step_stat(2)
        num_samples = 20
        cls.sem = StationaryModel()
        x_eps_0 = tfp.distributions.Normal(0.0, 1.0).sample((num_samples, 1))
        x_eps_1_2 = tfp.distributions.Normal(0.0, 0.001).sample((num_samples, 2))
        x_eps = tf.concat([x_eps_0, x_eps_1_2], axis=1)
        epsilon = OrderedDict(
            [
                ("X", x_eps),
                ("Z", tfp.distributions.Normal(0.0, 0.001).sample((num_samples, 3))),
                ("Y", tfp.distributions.Normal(0.0, 0.001).sample((num_samples, 3))),
            ]
        )
        cls.D_obs = draw_samples_from_sem(cls.sem, num_samples, 3, epsilon=epsilon)
        cls.fcns_full = fcns4sem(cls.tested_graph.graph, cls.D_obs)
        cls.sem_estimated = sem_hat(cls.fcns_full)()

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for utils.sem_estimate")

    def test_fcns4sem(self):
        self.assertEqual(len(self.fcns_full), 3)
        self.assertEqual(len(self.fcns_full[0]), 3)
        self.assertEqual(self.fcns_full[1].keys(), {"X", "Z", "Y"})
        self.assertIsInstance(self.fcns_full[2], OrderedDict)
        self.assertIsInstance(self.fcns_full[0]["X"](0), tf.Tensor)

        fcns = fcns4sem(self.tested_graph.graph, self.D_obs, temporal_index=1)
        self.assertEqual(len(fcns), 3)
        self.assertEqual(len(fcns[0]), 0)
        self.assertEqual(len(fcns[1]), 3)
        self.assertEqual(len(fcns[2]), 0)

    def test_sem_hat(self):
        the_sample = OrderedDict(
            [(key, []) for key in self.sem_estimated.static().keys()]
        )
        static_fcns = self.sem_estimated.static()

        # test static
        for key in self.sem_estimated.static().keys():
            the_sample[key].append(static_fcns[key](the_sample))
        self.assertEqual(len(the_sample), 3)
        self.assertEqual(len(the_sample["Z"]), 1)
        self.assertIsInstance(the_sample["Y"][0], tf.Tensor)

        # test dynamic
        dynamic_fcns = self.sem_estimated.dynamic()
        for t in [1, 2]:
            for key in self.sem_estimated.dynamic().keys():
                the_sample[key].append(dynamic_fcns[key](t, the_sample))

            self.assertEqual(len(the_sample), 3)
            self.assertEqual(len(the_sample["Z"]), t + 1)
            self.assertIsInstance(the_sample["Y"][t], tf.Tensor)

        for key in the_sample.keys():
            the_sample[key] = tf.reshape(tf.convert_to_tensor(the_sample[key]), (1, -1))

        # test correctness of the SEM estimation
        intervention = {
            "X": [the_sample["X"][0, 0], None, None],
            "Z": [None, None, None],
            "Y": [None, None, None],
        }
        epsilon = OrderedDict(
            [
                ("X", tf.constant([0.0, 0.0, 0.0])),
                ("Z", tf.constant([0.0, 0.0, 0.0])),
                ("Y", tf.constant([0.0, 0.0, 0.0])),
            ]
        )
        true_sample = sample_from_sem(
            self.sem, 3, intervention=intervention, epsilon=epsilon
        )

        self.assertTrue(
            tf.experimental.numpy.allclose(
                the_sample["X"], true_sample["X"], rtol=1e-1, atol=1e-1, equal_nan=False
            ),
        )
        self.assertTrue(
            tf.experimental.numpy.allclose(
                the_sample["Z"], true_sample["Z"], rtol=1e-1, atol=1e-1, equal_nan=False
            ),
        )
        self.assertTrue(
            tf.experimental.numpy.allclose(
                the_sample["Y"], true_sample["Y"], rtol=1e-1, atol=1e-1, equal_nan=False
            ),
        )

    def test_sample_from_sem_hat(self):

        # test the sample from the estimated SEM without intervention
        the_sample = sample_from_sem_hat(self.sem_estimated, 3)
        self.assertEqual(len(the_sample), 3)
        self.assertEqual(the_sample["Z"].shape, (3,))
        self.assertIsInstance(the_sample["Y"][0], tf.Tensor)

        # test the sample from the estimated SEM with intervention
        intervention = {
            "X": [the_sample["X"][0], None],
            "Z": [None, None],
            "Y": [None, None],
        }
        interven_sample = sample_from_sem_hat(
            self.sem_estimated, 2, intervention=intervention
        )

        self.assertEqual(len(interven_sample), 3)
        self.assertIsInstance(interven_sample["Y"][0], tf.Tensor)
        self.assertEqual(interven_sample["Z"].shape, (2,))
        self.assertEqual(interven_sample["X"][0], the_sample["X"][0])

    def test_draw_samples_from_sem_hat(self):
        intervention = {
            "X": [None, None, None],
            "Z": [1.3, None, None],
            "Y": [None, None, None],
        }
        num_samples = 20

        samples = draw_samples_from_sem_hat(
            self.sem_estimated, num_samples, 3, intervention=intervention
        )

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples["Y"].shape, (num_samples, 3))
        self.assertEqual(samples["Z"][10, 0], 1.3)

    def test_fy_and_fny(self):
        fy_fcns, fny_fcns = fy_and_fny(
            self.tested_graph.graph, self.D_obs, target_node_name="Y", temporal_index=1
        )

        ipt_fy = tf.expand_dims(tf.linspace(-3.0, 3.0, 100), axis=1)
        self.assertEqual(fy_fcns[0](ipt_fy).shape, (100,))
        self.assertEqual(fy_fcns[1](ipt_fy).shape, (100,))
        self.assertEqual(fny_fcns[0](ipt_fy).shape, (100,))
        self.assertEqual(fny_fcns[1](ipt_fy).shape, (100,))

        fy_fcns, fny_fcns = fy_and_fny(
            self.tested_graph.graph, self.D_obs, target_node_name="Y"
        )
        self.assertEqual(len(fy_fcns), 3)
        self.assertEqual(fy_fcns[0][0], None)
        self.assertEqual(len(fy_fcns[1]), 2)
        self.assertEqual(len(fny_fcns[0]), 2)
        self.assertEqual(len(fny_fcns[1]), 2)

        fy_fcns, fny_fcns = fy_and_fny(
            self.tested_graph.graph, self.D_obs, target_node_name="Y", temporal_index=0
        )
        self.assertEqual(fy_fcns, [None, None])
