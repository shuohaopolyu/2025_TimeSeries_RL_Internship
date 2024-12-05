from graphs.causalgraph import CausalGraph
import networkx as nx
import matplotlib.pyplot as plt
from utils.sequential_sampling import draw_samples_from_sem
from equations.stationary import StationaryModel
import tensorflow as tf
from collections import OrderedDict
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import time

time_start = time.time()
sem = StationaryModel()
intervention = {
    "X": [2., None, None],
    "Z": [None, -2., None],
    "Y": [None, None, None],
}
samples_1 = draw_samples_from_sem(sem, num_samples=1000, max_time_step=3, intervention=intervention)
time_end = time.time()
print('time cost', time_end-time_start, 's')

time_start = time.time()
sem = StationaryModel()
intervention = {
    "X": [2., None, None],
    "Z": [None, -2., None],
    "Y": [None, None, None],
}
epsilon = OrderedDict([("X", tf.zeros([1, 3])), ("Z", tf.zeros([1, 3])), ("Y", tf.zeros([1, 3]))])
samples_2 = draw_samples_from_sem(sem, num_samples=1, max_time_step=3, intervention=intervention, epsilon=epsilon)
time_end = time.time()
print('time cost', time_end-time_start, 's')

sns.displot(samples_1["Y"][:, 1], kde=True)
plt.axvline(samples_2["Y"][:, 1], color='r', linestyle='dashed', linewidth=1)
plt.axvline(tf.reduce_mean(samples_1["Y"][:, 1]), color='k', linestyle='dashed', linewidth=1)
plt.show()


# vertices = ['A', 'B', 'C', 'D', 'E']
# edges = [('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E')]
# cgobj = CausalGraph(vertices, edges, ['A', 'B', 'C', 'D'], 'E')
# cgobj.minimal_interven_set()

#vertices is from A to O, plus Y
# vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Y']
# edges = [ ('A', 'Y'), 
#          ('B', 'M'), ('B', 'L'), 
#          ('C', 'A'), ('C', 'B'), ('C', 'I'),
#          ('D', 'H'), ('D', 'Y'),
#          ('E', 'D'), ('E', 'Y'),
#          ('F', 'E'),
#          ('G', 'F'), ('G', 'L'), 
#          ('I', 'A'), ('I', 'J'),
#          ('J', 'D'), ('J', 'H'), 
#          ('K', 'Y'), ('K', 'B'), ('K', 'C'),
#          ('L', 'Y'), ('L', 'F'),
#          ('M', 'N'), ('M', 'G'), ('M', 'L'),
#          ('N', 'G')]
# treat_vars =[ 'A', 'B', 'C', 'D', 'E', 'F', 'G']
# cgobj = CausalGraph(vertices, edges, treat_vars, 'Y')
# nx.draw(cgobj.graph, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
# plt.show()
# print(len(cgobj.minimal_interven_set()))

