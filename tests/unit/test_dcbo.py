import unittest
from methods.dcbo import DynCausalBayesOpt
from causal_graph.example_dyn_graphs import three_step_stat
from sem.stationary import StationaryModel, StationaryModel_dev
from utils.sequential_sampling import draw_samples_from_sem, draw_samples_from_sem_dev
from collections import OrderedDict
import tensorflow as tf
import math


class TestDynCausalBayesOpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for methods.dcbo")
        dyn_graph = three_step_stat(0)
        sem = StationaryModel_dev()
        D_obs = draw_samples_from_sem_dev(sem, 20, 0)

        intervention = {
            "X": [0.5],
            "Z": [None],
            "Y": [None],
        }
        D_intervene_X_1 = draw_samples_from_sem_dev(
            sem, 1, 0, intervention=intervention, epsilon=0.0
        )

        intervention = {
            "X": [-2.1],
            "Z": [None],
            "Y": [None],
        }
        D_intervene_X_2 = draw_samples_from_sem_dev(
            sem, 1, 0, intervention=intervention, epsilon=0.0
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
        D_intervene_Z = draw_samples_from_sem_dev(
            sem, 1, 0, intervention=intervention, epsilon=0.0
        )

        # with a zero epsilon, the samples are deterministic, thus monte carlo is not needed
        D_intervene_ini = OrderedDict(
            [(("X",), D_intervene_X), (("Z",), D_intervene_Z)]
        )
        intervention_domain = OrderedDict([("X", [-5.0, 5.0]), ("Z", [-5.0, 20.0])])
        cls.dcbo_stat = DynCausalBayesOpt(
            dyn_graph,
            sem,
            D_obs,
            D_intervene_ini,
            intervention_domain,
            num_trials=10,
            task="min",
            num_monte_carlo=100,
            num_anchor_points=5,
        )

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for methods.dcbo")

    def test_initialization(self):
        self.assertEqual(
            self.dcbo_stat.exploration_set, [("X",), ("Z",)]
        )  # _initialize_exploration_set
        self.assertEqual(self.dcbo_stat.T, 3)
        self.assertEqual(self.dcbo_stat.target_var, "Y")

    def test_optimal_intervene_value(self):
        es, es_values, global_extreme = self.dcbo_stat._optimal_intervene_value(0)
        self.assertEqual(es, ("Z",))
        self.assertTrue(math.isclose(es_values[0], -3.1, abs_tol=1e-6))
        self.assertEqual(
            global_extreme, self.dcbo_stat.D_interven[0][("Z",)]["Y"][0, 0].numpy()
        )

    def test_intervene_scheme(self):
        self.dcbo_stat._update_opt_intervene_history(0)
        x_py = ["X_1"]
        i_py = []
        x_py_values = tf.constant([[-3.2], [4.2]])
        output_intervention = self.dcbo_stat._intervene_scheme(x_py, i_py, x_py_values)
        true_intervention = {
            "X": [None, -3.2],
            "Z": [None, None],
            "Y": [None, None],
        }
        self.assertEqual(output_intervention[0], true_intervention)

    def test_update_opt_intervene_vars(self):
        self.dcbo_stat._update_opt_intervene_history(0)
        self.dcbo_stat._update_opt_intervene_vars(0)
        self.assertEqual(self.dcbo_stat.full_opt_intervene_vars, ["Z_0"])

    def test_input_fny(self):
        samples = draw_samples_from_sem_dev(self.dcbo_stat.sem, 100, 2)
        self.dcbo_stat.dyn_graph.temporal_index = 2
        the_graph = self.dcbo_stat.dyn_graph.graph
        ipt = self.dcbo_stat._input_fny(the_graph, samples)
        self.assertEqual(ipt.shape, (100, 1))
        self.assertEqual(ipt[0, 0], samples["Z"][0, 2])

    def test_intervention_points(self):
        for es in self.dcbo_stat.exploration_set:
            self.assertIsInstance(self.dcbo_stat._intervention_points(es), tf.Tensor)
            self.assertEqual(
                self.dcbo_stat._intervention_points(es).shape,
                (self.dcbo_stat.num_anchor_points, 1),
            )

    def test_prior_causal_gp(self):
        D_obs = draw_samples_from_sem_dev(self.dcbo_stat.sem, 20, 0)
        self.dcbo_stat.D_obs = D_obs
        self.dcbo_stat._prior_causal_gp(0)
        D_obs = draw_samples_from_sem_dev(self.dcbo_stat.sem, 20, 1)
        self.dcbo_stat.D_obs = D_obs
        self.dcbo_stat._update_opt_intervene_history(0)
        self.dcbo_stat._update_sem_hat(1)
        mean_std_es = self.dcbo_stat._prior_causal_gp(1)
        mean_z = mean_std_es[("Z",)][0]
        std_z = mean_std_es[("Z",)][1]
        example_input = tf.constant([[1.2], [2.3]])
        pred_mean_z = mean_z(example_input)
        pred_std_z = std_z(example_input)
        self.assertIsInstance(pred_mean_z, tf.Tensor)
        self.assertIsInstance(pred_std_z, tf.Tensor)
        self.assertEqual(pred_mean_z.shape, (2, ))
        self.assertEqual(pred_std_z.shape, (2, ))

    def test_posterior_causal_gp(self):
        D_obs = draw_samples_from_sem_dev(self.dcbo_stat.sem, 20, 1)
        self.dcbo_stat.D_obs = D_obs
        self.dcbo_stat._update_opt_intervene_history(0)
        self.dcbo_stat._update_sem_hat(1)
        self.dcbo_stat._prior_causal_gp(1)

        intervention = {
            "X": [None, 3.2],
            "Z": [None, None],
            "Y": [None, None],
        }
        D_intervene_X = draw_samples_from_sem_dev(
            self.dcbo_stat.sem, 1, 1, intervention=intervention, epsilon=0.0
        )

        self.dcbo_stat.D_interven[1] = OrderedDict([(("X", ), D_intervene_X)])
        mean_std_es = self.dcbo_stat._posterior_causal_gp(1)
        mean_es = mean_std_es[("X",)][0]
        candidate_point = tf.constant([[3.2], [4.2], [5.2]])
        candidate_point = tf.expand_dims(candidate_point, axis=0)
        pred_mean = mean_es(candidate_point)
        self.assertEqual(pred_mean.shape, (1,3))
        self.dcbo_stat._acquisition_function(1)

    def test_suspected_intervention_this_trial(self):
        candidates_x = tf.constant([[-5.0], [-2.5], [0.0], [2.5], [5.0]])
        candidates_z = tf.constant([[-5.0], [1.25], [7.5], [13.75], [20.0]])
        aq_x = tf.constant(
            [[-1.0522499, -1.0522504, -1.0524182, -1.0689896, -1.057488]]
        )
        aq_z = tf.constant(
            [[-1.0522499, -1.0562664, -1.0522525, -1.0522499, -1.0522499]]
        )
        self.dcbo_stat.D_acquisition[("X",)] = [candidates_x, aq_x]
        self.dcbo_stat.D_acquisition[("Z",)] = [candidates_z, aq_z]
        suspected_es, suspected_candidate_point = (
            self.dcbo_stat._suspected_intervention_this_trial()
        )
        self.assertEqual(suspected_es, ("X",))
        self.assertTrue(
            math.isclose(suspected_candidate_point[0, 0], 2.5, abs_tol=1e-6)
        )
        self.assertEqual(suspected_candidate_point.shape, (1, 1))

    def test_intervene_and_augment(self):
        suspected_es = ("X",)
        suspected_candidate_point = tf.constant([[2.5]])
        self.dcbo_stat._intervene_and_augment(0, suspected_es, suspected_candidate_point)
        self.assertEqual(self.dcbo_stat.D_interven[0][("X",)]["Y"].shape, (3, 1))
        self.dcbo_stat._update_opt_intervene_history(0)
        self.dcbo_stat._intervene_and_augment(1, suspected_es, suspected_candidate_point)
        self.assertEqual(self.dcbo_stat.D_interven[1][("X",)]["Y"].shape, (1, 2))

