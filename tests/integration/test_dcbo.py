from methods.dcbo import DynCausalBayesOpt
from utils.sequential_sampling import draw_samples_from_sem_dev
from causal_graph.example_dyn_graphs import three_step_stat
from sem.stationary import StationaryModel_dev
from collections import OrderedDict
from utils.costs import equal_cost
import unittest


class TestDynCausalBayesOpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting integration tests for methods.dcbo")
        dyn_graph = three_step_stat()
        sem_model = StationaryModel_dev()
        D_obs = draw_samples_from_sem_dev(sem_model, 20, 2, seed=1111)
        intervention_ini = {
            "X": [-3.6],
            "Z": [None],
            "Y": [None],
        }
        D_intervene_ini_x = draw_samples_from_sem_dev(
            sem_model, 1, 0, intervention=intervention_ini, epsilon=0.0
        )
        intervention_ini = {
            "X": [None],
            "Z": [12.5],
            "Y": [None],
        }
        D_intervene_ini_z = draw_samples_from_sem_dev(
            sem_model, 1, 0, intervention=intervention_ini, epsilon=0.0
        )
        D_intervene_ini = OrderedDict(
                    [(("X",), D_intervene_ini_x), (("Z",), D_intervene_ini_z)]
                )
        intervention_domain = OrderedDict([("X", [-5.0, 5.0]), ("Z", [-5.0, 20.0])])
        num_trials = 2
        task = "min"
        cost_fn = equal_cost
        num_anchor_points = 100
        num_monte_carlo = 1000
        jitter = 1e-6
        cls.dcbo = DynCausalBayesOpt(
            dyn_graph,
            sem_model,
            D_obs,
            D_intervene_ini,
            intervention_domain,
            num_trials,
            task,
            cost_fn,
            num_anchor_points,
            num_monte_carlo,
            jitter,
        )

    @classmethod
    def tearDownClass(cls):
        print("Finished integration tests for methods.dcbo")

    def test_run(self):
        self.dcbo.run()
        self.assertTrue(True)