import unittest
from methods.dcbo import DynCausalBayesOpt
from causal_graph.dynamic_graph import DynCausalGraph
from sem.stationary import StationaryModel


class TestDynCausalBayesOpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for methods.dcbo")
        full_vertices = ["X_0", "Z_0", "Y_0", "X_1", "Z_1", "Y_1", "X_2", "Z_2", "Y_2"]
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
        ]
        full_treat_vars = [["X_0", "Z_0"], ["X_1", "Z_1"], ["X_2", "Z_2"]]
        full_output_vars = ["Y_0", "Y_1", "Y_2"]
        full_do_vars = [[], [], []]
        temporal_index = 0
        dyn_graph = DynCausalGraph(
            full_vertices,
            full_edges,
            full_treat_vars,
            full_output_vars,
            full_do_vars,
            temporal_index,
        )
        sem = StationaryModel()
        cls.dcbo_stat = DynCausalBayesOpt(dyn_graph, sem, num_trials=10, task="min")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for methods.dcbo")

    def test_initialize_exploration_set(self):
        exploration_set = self.dcbo_stat._initialize_exploration_set(0)
        self.assertEqual(exploration_set,  [["X_0"], ["Z_0"]])
        exploration_set = self.dcbo_stat._initialize_exploration_set(1)
        self.assertEqual(exploration_set,  [["X_1"], ["Z_1"]])
        exploration_set = self.dcbo_stat._initialize_exploration_set(2)
        self.assertEqual(exploration_set,  [["X_2"], ["Z_2"]])

    def test_optimize(self):
        pass
