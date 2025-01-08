import unittest
import tensorflow as tf
import tensorflow_probability as tfp
from utils.sem_estimate import fcns4sem, sem_hat, fy_and_fny
from causal_graph.dynamic_graph import DynCausalGraph
from causal_graph.example_dyn_graphs import three_step_stat

from utils.sequential_sampling import (
    draw_samples_from_sem_dev,
    draw_samples_from_sem_hat_dev,
)
from sem.stationary import StationaryModel, StationaryModel_dev
from collections import OrderedDict



class TestSemEstimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for utils.sem_estimate")
        cls.tested_graph = three_step_stat()
        cls.tested_graph.temporal_index = 1
        num_samples = 20
        cls.sem = StationaryModel_dev()
        x_eps_0 = tfp.distributions.Normal(0.0, 1.0).sample((num_samples, 1))
        x_eps_1 = tfp.distributions.Normal(0.0, 0.1).sample((num_samples, 1))
        x_eps = tf.concat([x_eps_0, x_eps_1], axis=1)
        epsilon = OrderedDict(
            [
                ("X", x_eps),
                ("Z", tfp.distributions.Normal(0.0, 0.1).sample((num_samples, 2))),
                ("Y", tfp.distributions.Normal(0.0, 0.1).sample((num_samples, 2))),
            ]
        )
        cls.D_obs = draw_samples_from_sem_dev(cls.sem, num_samples, 1, epsilon=epsilon)
        cls.fcns_full = fcns4sem(cls.tested_graph.graph, cls.D_obs, debug_mode=False)
        cls.sem_estimated = sem_hat(cls.fcns_full)()
        print("Finished setting up tests for utils.sem_estimate")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for utils.sem_estimate")

    def test_fcns4sem(self):
        self.assertEqual(len(self.fcns_full), 2)
        self.assertEqual(len(self.fcns_full[0]), 3)
        self.assertEqual(self.fcns_full[1].keys(), {"X", "Z", "Y"})
        self.assertIsInstance(self.fcns_full[0]["X"](0), tf.Tensor)
        fcns = fcns4sem(self.tested_graph.graph, self.D_obs, temporal_index=1, debug_mode=False)
        self.assertEqual(len(fcns), 2)
        print(fcns[0])
        self.assertEqual(len(fcns[0]), 0)
        self.assertEqual(len(fcns[1]), 3)

    def test_draw_samples_from_sem_hat_dev(self):
        intervention = {
            "X": [None, None],
            "Z": [1.3, None],
            "Y": [None, None],
        }
        num_samples = 10000
        samples = draw_samples_from_sem_hat_dev(
            self.sem_estimated, num_samples, 1, intervention=intervention
        )
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples["Y"].shape, (num_samples, 2))
        self.assertEqual(samples["Z"][4, 0], 1.3)


