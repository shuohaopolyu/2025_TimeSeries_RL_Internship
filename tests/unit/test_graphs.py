import unittest
import graphs.utils as utils
import graphs.CasualGraph as cg


class TestUtils(unittest.TestCase):
    def test_is_subset(self):
        self.assertTrue(utils.is_subset(['A', 'B', 'C'], ['A', 'B', 'C', 'D', 'E']))
        self.assertFalse(utils.is_subset(['A', 'B', 'C'], ['A', 'B', 'D', 'E']))


class TestCasualGraph(unittest.TestCase):
    def setUp(self):
        self.tested_graph = cg.CasualGraph(
            ['A', 'B', 'C', 'D', 'E'],
            [('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E')],
            ['A', 'C', 'D'],
            'E',
        )
        self.assertEqual(self.tested_graph.treat_vars, ['A', 'C', 'D'])
        self.assertEqual(self.tested_graph.output_var, 'E')
        self.assertEqual(self.tested_graph.non_manip_vars, ['B'])

    def test_do(self):
        self.assertEqual(list(self.tested_graph.graph.nodes), ['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(list(self.tested_graph.do(['C', 'D']).nodes), ['C', 'D', 'E'])
        self.assertEqual(list(self.tested_graph.do(['A', 'C']).nodes), ['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(list(self.tested_graph.do(['A']).nodes), ['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(list(self.tested_graph.do(['A', 'C', 'D']).nodes), ['C', 'D', 'E'])

    def test_in_do_graph(self):
        self.assertTrue(self.tested_graph.in_do_graph(['C', 'D']))
        self.assertFalse(self.tested_graph.in_do_graph(['A', 'C', 'D']))
        self.assertTrue(self.tested_graph.in_do_graph(['A', 'C']))

    def test_power_set(self):
        self.assertEqual(self.tested_graph.power_set(['A', 'B', 'C']), [[], ['C'], ['B'], ['B', 'C'], ['A'], ['A', 'C'], ['A', 'B'], ['A', 'B', 'C']])

if __name__ == '__main__':
    unittest.main()
