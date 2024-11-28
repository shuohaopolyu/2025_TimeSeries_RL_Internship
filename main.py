import graphs.CasualGraph as cg
import networkx as nx
import matplotlib.pyplot as plt

vertices = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E')]
cgobj = cg.CasualGraph(vertices, edges, ['A', 'B', 'C', 'D'], 'E')
cgobj.mis()

# # Draw the graph
# nx.draw(cgobj.graph, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
# plt.show()

# do_graph = cgobj.do(['C', 'D'])
# print(cgobj.do(['C', 'D']).nodes)
# # Draw the do_graph
# nx.draw(do_graph, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
# plt.show()