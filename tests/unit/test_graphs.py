import unittest
import graphs.utils as utils
from graphs.causalgraph import CausalGraph


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for graphs.utils")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for graphs.utils")

    def test_is_subset(self):
        self.assertTrue(utils.is_subset(["A", "B", "C"], ["A", "B", "C", "D", "E"]))
        self.assertFalse(utils.is_subset(["A", "B", "C"], ["A", "B", "D", "E"]))

    def test_power_set(self):
        self.assertEqual(
            utils.power_set(["A", "B", "C"]),
            [
                [],
                ["C"],
                ["B"],
                ["B", "C"],
                ["A"],
                ["A", "C"],
                ["A", "B"],
                ["A", "B", "C"],
            ],
        )


class TestCausalGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting tests for graphs.CausalGraph")

    @classmethod
    def tearDownClass(cls):
        print("Finished tests for graphs.CausalGraph")

    def test_init(self):

        # test 1: normal initialization, using exmple A->B->C
        self.tested_graph = CausalGraph(
            ["A", "B", "C"], [("A", "B"), ("B", "C")], ["A"], "C"
        )
        self.assertEqual(self.tested_graph.treat_vars, ["A"])
        self.assertEqual(self.tested_graph.output_var, "C")
        self.assertEqual(self.tested_graph.non_manip_vars, ["B"])

        # test 2: output_var not in vertices
        with self.assertRaises(AssertionError):
            self.tested_graph = CausalGraph(
                ["A", "B", "C"], [("A", "B"), ("B", "C")], ["A"], "D"
            )

        # test 3: treat_vars not in vertices
        with self.assertRaises(AssertionError):
            self.tested_graph = CausalGraph(
                ["A", "B", "C"], [("A", "B"), ("B", "C")], ["A", "D"], "C"
            )

        # test 4: output_var in treat_vars
        with self.assertRaises(AssertionError):
            self.tested_graph = CausalGraph(
                ["A", "B", "C"], [("A", "B"), ("B", "C")], ["A", "C"], "C"
            )

        # test 5: graph is not a DAG
        with self.assertRaises(AssertionError):
            self.tested_graph = CausalGraph(
                ["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")], ["A"], "C"
            )

    def test_do(self):
        # test 1: normal do, using exmple A->B->C->D
        self.tested_graph = CausalGraph(
            ["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D")], ["B", "C"], "D"
        )
        self.assertEqual(list(self.tested_graph.graph.nodes), ["A", "B", "C", "D"])
        self.assertEqual(list(self.tested_graph.do(["B"]).nodes), ["B", "C", "D"])
        self.assertEqual(list(self.tested_graph.do(["B", "C"]).nodes), ["C", "D"])
        self.assertEqual(list(self.tested_graph.do(["C"]).nodes), ["C", "D"])

        # test 2: do_vars not in treat_vars
        with self.assertRaises(AssertionError):
            self.tested_graph.do(["A"])

    def test_in_do_graph(self):
        # using exmple A->B->C->D
        self.tested_graph = CausalGraph(
            ["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D")], ["B", "C"], "D"
        )
        self.assertTrue(self.tested_graph.in_do_graph(["B"]))
        self.assertFalse(self.tested_graph.in_do_graph(["B", "C"]))
        self.assertTrue(self.tested_graph.in_do_graph(["C"]))

    def test_MIS(self):
        # using exmple A->B->C
        self.tested_graph = CausalGraph(
            ["A", "B", "C"], [("A", "B"), ("B", "C")], ["A", "B"], "C"
        )
        self.assertEqual(self.tested_graph.minimal_interven_set().sort(), [[], ["B"], ["A"]].sort())

        # using example A_1->A_2->...->A_10->Y
        vertices = [f"A_{i}" for i in range(1, 11)] + ["Y"]
        edges = [(f"A_{i}", f"A_{i+1}") for i in range(1, 10)] + [(f"A_10", "Y")]
        treat_vars = [f"A_{i}" for i in range(1, 11)]
        self.tested_graph = CausalGraph(vertices, edges, treat_vars, "Y")
        self.assertEqual(len(self.tested_graph.minimal_interven_set()), 11)

        # using example A1->A2->A3->A4->A5->Y and A6->A7->A8->A9->A10->Y
        vertices = [f"A_{i}" for i in range(1, 11)] + ["Y"]
        edges = [(f"A_{i}", f"A_{i+1}") for i in range(1, 5)] + [(f"A_5", "Y")]
        edges += [(f"A_{i}", f"A_{i+1}") for i in range(6, 10)] + [(f"A_10", "Y")]
        treat_vars = [f"A_{i}" for i in range(1, 11)]
        self.tested_graph = CausalGraph(vertices, edges, treat_vars, "Y")
        self.assertEqual(len(self.tested_graph.minimal_interven_set()), 36)


if __name__ == "__main__":
    unittest.main()
