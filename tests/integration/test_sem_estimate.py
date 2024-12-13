import unittest
import tensorflow as tf
import tensorflow_probability as tfp
from utils.sem_estimate import fcns4sem, sem_hat
from causal_graph.dynamic_graph import DynCausalGraph
from utils.sequential_sampling import draw_samples_from_sem
from sem.stationary import StationaryModel
from collections import OrderedDict

class TestSemEstimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for utils.sem_estimate")
        cls.tested_graph = DynCausalGraph(
            full_vertices = ["X_0", "X_1", "X_2", "Z_0", "Z_1", "Z_2", "Y_0", "Y_1", "Y_2"],
            full_edges = [
                ("X_0", "Z_0"),
                ("Z_0", "Y_0"),
                ("X_1", "Z_1"),
                ("Z_1", "Y_1"),
                ("X_2", "Z_2"),
                ("Z_2", "Y_2"),
                ("X_0", "X_1"),
                ("X_1", "X_2"),
                ("Z_0", "Z_1"),
                ("Z_1", "Z_2"),
                ("Y_0", "Y_1"),
                ("Y_1", "Y_2"),
            ],
            full_treat_vars = [["X_0", "Z_0"], ["X_1", "Z_1"], ["X_2", "Z_2"]],
            full_do_vars = [[], [], []],
            full_output_vars = ["Y_0", "Y_1", "Y_2"],
            temporal_index = 2,
        )
        sem = StationaryModel()
        cls.D_obs = draw_samples_from_sem(sem, 10, 3)

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for utils.sem_estimate")

    def test_fcns4sem(self):
        fcns = fcns4sem(self.tested_graph, self.D_obs)
        self.assertEqual(len(fcns), 3)
        self.assertEqual(len(fcns[0]), 3)
        self.assertEqual(fcns[1].keys(), {"X", "Z", "Y"})
        self.assertIsInstance(fcns[2], OrderedDict)
        self.assertIsInstance(fcns[0]["X"](0), tf.Tensor)

    def test_sem_hat(self):
        sem = sem_hat(self.tested_graph, self.D_obs)()
        the_sample = OrderedDict([(key, []) for key in sem.static().keys()])
        static_fcns = sem.static()

        # test static
        for key in sem.static().keys():
            the_sample[key].append(static_fcns[key](the_sample))
        self.assertEqual(len(the_sample), 3)
        self.assertEqual(len(the_sample["Z"]), 1)
        self.assertIsInstance(the_sample["Y"][0], tf.Tensor)

        # test dynamic
        dynamic_fcns = sem.dynamic()
        for t in [1, 2]:
            for key in sem.dynamic().keys():
                the_sample[key].append(dynamic_fcns[key](t, the_sample))
            self.assertEqual(len(the_sample), 3)
            self.assertEqual(len(the_sample["Z"]), t + 1)
            self.assertIsInstance(the_sample["Y"][t], tf.Tensor)


