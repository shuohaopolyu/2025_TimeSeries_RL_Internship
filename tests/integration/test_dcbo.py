import unittest
from methods.dcbo import DynCausalBayesOpt
from causal_graph.dynamic_graph import DynCausalGraph
from causal_graph.example_dyn_graphs import three_step_stat
from sem.stationary import StationaryModel
from utils.sequential_sampling import draw_samples_from_sem
from collections import OrderedDict


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

        # with a zero epsilon, the samples are deterministic, thus monte carlo is not needed
        D_interven_ini = OrderedDict([("X", draw_samples_from_sem(sem, 1, 1, intervention=intervention, epsilon=0.0))])

        intervention_domain = OrderedDict([("X", [-5.0, 5.0]), ("Z", [-5.0, 20.0])])
        cls.dcbo_stat = DynCausalBayesOpt(
            dyn_graph,
            sem,
            D_obs,
            D_interven_ini,
            intervention_domain,
            num_trials=10,
            task="min",
        )

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for methods.dcbo")

    def test_initialization(self):
        self.assertEqual(self.dcbo_stat.exploration_set, [["X"], ["Z"]]) # _initialize_exploration_set
        self.assertEqual(self.dcbo_stat.T, 3)
        self.assertEqual(self.dcbo_stat.target_var, "Y")

    def test_optimal_interven_value(self):
        es, es_values, global_extreme = self.dcbo_stat._optimal_interven_value(0)
        self.assertEqual(es, ("X"))
        self.assertEqual(es_values, [0.5])
        self.assertEqual(global_extreme, self.dcbo_stat.D_interven[0]["X"]["Y"][0,0].numpy())
