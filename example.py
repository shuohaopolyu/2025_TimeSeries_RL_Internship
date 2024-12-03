import graphs.CasualGraph as cg
import networkx as nx
import matplotlib.pyplot as plt

# vertices = ['A', 'B', 'C', 'D', 'E']
# edges = [('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E')]
# cgobj = cg.CasualGraph(vertices, edges, ['A', 'B', 'C', 'D'], 'E')
# cgobj.MIS()

#vertices is from A to O, plus Y
vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Y']
edges = [ ('A', 'Y'), 
         ('B', 'M'), ('B', 'L'), 
         ('C', 'A'), ('C', 'B'), ('C', 'I'),
         ('D', 'H'), ('D', 'Y'),
         ('E', 'D'), ('E', 'Y'),
         ('F', 'E'),
         ('G', 'F'), ('G', 'L'), 
         ('I', 'A'), ('I', 'J'),
         ('J', 'D'), ('J', 'H'), 
         ('K', 'Y'), ('K', 'B'), ('K', 'C'),
         ('L', 'Y'), ('L', 'F'),
         ('M', 'N'), ('M', 'G'), ('M', 'L'),
         ('N', 'G')]
treat_vars =[ 'A', 'B', 'C', 'D', 'E', 'F', 'G']
cgobj = cg.CasualGraph(vertices, edges, treat_vars, 'Y')
nx.draw(cgobj.graph, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
plt.show()
print(len(cgobj.MIS()))

# # Draw the graph
# nx.draw(cgobj.graph, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
# plt.show()
# do_graph = cgobj.do(['C', 'D'])
# print(cgobj.do(['C', 'D']).nodes)
# # Draw the do_graph
# nx.draw(do_graph, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
# plt.show()