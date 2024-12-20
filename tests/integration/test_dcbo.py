import unittest
from methods.dcbo import DynCausalBayesOpt
from causal_graph.example_dyn_graphs import three_step_stat
from sem.stationary import StationaryModel
from utils.sequential_sampling import draw_samples_from_sem
from collections import OrderedDict
import tensorflow as tf
import math


class TestDynCausalBayesOpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for methods.dcbo")
        dyn_graph = three_step_stat(0)
        sem = StationaryModel()
        D_obs = draw_samples_from_sem(sem, 4, 3)

        intervention = {
            "X": [0.5],
            "Z": [None],
            "Y": [None],
        }
        D_intervene_X_1 = draw_samples_from_sem(
            sem, 1, 1, intervention=intervention, epsilon=0.0
        )

        intervention = {
            "X": [-2.1],
            "Z": [None],
            "Y": [None],
        }
        D_intervene_X_2 = draw_samples_from_sem(
            sem, 1, 1, intervention=intervention, epsilon=0.0
        )
        D_intervene_X = OrderedDict()

        for key in D_intervene_X_1.keys():
            D_intervene_X[key] = tf.concat(
                [D_intervene_X_1[key], D_intervene_X_2[key]], axis=0
            )

        intervention = {
            "X": [None],
            "Z": [-3.1],
            "Y": [None],
        }
        D_intervene_Z = draw_samples_from_sem(
            sem, 1, 1, intervention=intervention, epsilon=0.0
        )

        # with a zero epsilon, the samples are deterministic, thus monte carlo is not needed
        D_intervene_ini = OrderedDict([("X", D_intervene_X), ("Z", D_intervene_Z)])
        intervention_domain = OrderedDict([("X", [-5.0, 5.0]), ("Z", [-5.0, 20.0])])
        cls.dcbo_stat = DynCausalBayesOpt(
            dyn_graph,
            sem,
            D_obs,
            D_intervene_ini,
            intervention_domain,
            num_trials=10,
            task="min",
        )

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for methods.dcbo")

    def test_initialization(self):
        self.assertEqual(
            self.dcbo_stat.exploration_set, [["X"], ["Z"]]
        )  # _initialize_exploration_set
        self.assertEqual(self.dcbo_stat.T, 3)
        self.assertEqual(self.dcbo_stat.target_var, "Y")

    def test_optimal_intervene_value(self):
        es, es_values, global_extreme = self.dcbo_stat._optimal_intervene_value(0)
        self.assertEqual(es, ("Z"))
        self.assertTrue(math.isclose(es_values[0], -3.1, abs_tol=1e-6))
        self.assertEqual(
            global_extreme, self.dcbo_stat.D_interven[0]["Z"]["Y"][0, 0].numpy()
        )

    def test_intervene_scheme(self):
        self.dcbo_stat._update_opt_intervene_history(0)
        x_py = ["X_1"]
        i_py = []
        x_py_values = [-3.0]
        output_intervention = self.dcbo_stat._intervene_scheme(x_py, i_py, x_py_values)
        true_intervention = {
            "X": [None, -3.0],
            "Z": [None, None],
            "Y": [None, None],
        }
        self.assertEqual(output_intervention, true_intervention)

    def test_update_opt_intervene_vars(self):
        self.dcbo_stat._update_opt_intervene_history(0)
        self.dcbo_stat._update_opt_intervene_vars(0)
        self.assertEqual(self.dcbo_stat.full_opt_intervene_vars, ["Z_0"])

    def test_input_fny(self):
        samples = draw_samples_from_sem(self.dcbo_stat.sem, 100, 3)
        self.dcbo_stat.dyn_graph.temporal_index = 2
        the_graph = self.dcbo_stat.dyn_graph.graph
        ipt = self.dcbo_stat._input_fny(the_graph, samples)
        self.assertEqual(ipt.shape, (100, 1))
        self.assertEqual(ipt[0, 0], samples["Z"][0, 2])

