import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from(
    [("US", "Israel"),
    ("AIPAC", "Israel"),
    ("Clinton", "AIPAC"),
    ("Clinton", "Obama"),
    ("Clinton", "US"),
    ("State", "Clinton"),
    ("Israel", "Clinton"),
    ("State", "Israel"),
    ("Clinton", "State"),
    ("Obama", "Israel"),
    ("AIPAC", "Clinton"),
    ("Obama", "Clinton"),
    ("US", "Clinton"),
    ("Clinton", "Israel")])

val_map = {'Isreal': 1.0,
           'State': 0.5714285714285714,
           'Clinton': 0.0,
           'US': 0.1,
           "AIPAC": 0.2,
           "Obama": 0.4}

values = [val_map.get(node, 0.25) for node in G.nodes()]
print(values)

# Specify the edges you want here
green_edges = [("US", "Israel"),
    ("AIPAC", "Israel"),
    ("Clinton", "AIPAC"),
    ("Clinton", "Obama"),
    ("Clinton", "US"),
    ("State", "Clinton"),
    ("Israel", "Clinton"),
    ("State", "Israel"),
    ("Clinton", "State"),
    ("Obama", "Israel"),
    ("AIPAC", "Clinton"),
    ("Obama", "Clinton"),
    ("US", "Clinton"),
    ("Clinton", "Israel")]
edge_colours = ['red' if not edge in green_edges else 'green'
                for edge in G.edges()]
red_edges = [edge for edge in G.edges() if edge not in green_edges]

# Need to create a layout when doing
# separate calls to draw nodes and edges
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                       node_size = 500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=green_edges, edge_color='g', arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
plt.show()