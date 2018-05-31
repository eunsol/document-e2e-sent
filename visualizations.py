import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import matplotlib.pyplot as plt
import json

'''
COLORS = ['blue', 'red', 'green', 'violet', 'orange',
          'yellow', 'black', 'brown', 'cyan', 'pink',
          'purple', 'magenta', 'lime', 'teal', 'olive',
          'teal', 'darkgray', 'lightgray', 'gray']
'''
# blue red green orange yellow
# fuschia/violet brown gray cyan maroon
# pink/magenta olive lime purple teal
# lightblue lightred lightgreen lightgray darkgray
COLORS = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (1, 0.5, 0), (1, 1, 0),
          (1, 0, 1), (0.4, 0.25, 0.13), (0.5, 0.5, 0.5), (0, 1, 1), (0.5, 0, 0),
          (1, 0.1, 0.5), (0.5, 0.5, 0), (0, 1, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
          (0.8, 0.8, 1), (1, 0.8, 0.8), (0.8, 1, 0.8), (0.8, 0.8, 0.8), (0.2, 0.2, 0.2)]
'''
TEXT_COLORS = ['black' for i in range(len(COLORS))]
TEXT_COLORS[6] = 'white'
'''

# either holder_inds and target_inds given
# or entity_inds given
def convert_to_str(tokens, holder_inds=None, target_inds=None, entity_inds=None):
    tokens = tokens.copy()
    if holder_inds is not None:
        for holder_ind in holder_inds:
            tokens[holder_ind[0]] = "\\textcolor{blue}{\\textbf{" + tokens[holder_ind[0]]
            tokens[holder_ind[1]] = tokens[holder_ind[1]] + "}}"
    if target_inds is not None:
        for target_ind in target_inds:
            tokens[target_ind[0]] = "\\textcolor{red}{\\textbf{" + tokens[target_ind[0]]
            tokens[target_ind[1]] = tokens[target_ind[1]] + "}}"
    if entity_inds is not None:
        for i in range(len(entity_inds)):
            for inds in entity_inds[i]:
                color_str = str(COLORS[i][0]) + ", " + str(COLORS[i][1]) + ", " + str(COLORS[i][2])
                tokens[inds[0]] = "\\textcolor[rgb]{" + color_str + "}{\\textbf{" + tokens[inds[0]]
                tokens[inds[1]] = tokens[inds[1]] + "}}"

    string = ""
    i = 0
    for token in tokens:
        if token is not "'" and token is not "." and token is not "\"\"" and token is not "," and token is not "''":
            string += " " + token
        else:
            string += token
        i += 1
    return string


all_data = []  # {}
doc_to_ht_to_annot = {}
doc_to_entities = {}
doc_to_entity_inds = {}
max_num_entities = 0
doc_to_ht_pairs_sentiment = {}
doc_to_ht_pairs_positive = {}
with open("./data/stanford_label_sent_boundary_with_label/acl_test.json", "r", encoding="latin1") as rf:
    for line in rf:
        annot = json.loads(line)
        if annot["docid"] not in doc_to_ht_to_annot:
            doc_to_ht_to_annot[annot["docid"]] = {}
            doc_to_ht_pairs_sentiment[annot["docid"]] = []
            doc_to_ht_pairs_positive[annot["docid"]] = []
            doc_to_entities[annot["docid"]] = {}
            doc_to_entity_inds[annot["docid"]] = []
        # Add entity
        entity = annot["holder"]
        if entity not in doc_to_entities[annot["docid"]]:
            doc_to_entities[annot["docid"]][entity] = 0
            doc_to_entity_inds[annot["docid"]].append(annot["holder_index"])
        entity = annot["target"]
        if entity not in doc_to_entities[annot["docid"]]:
            doc_to_entities[annot["docid"]][entity] = 0
            doc_to_entity_inds[annot["docid"]].append(annot["target_index"])
        # Update max # entities if necessary
        if len(doc_to_entities[annot["docid"]]) > max_num_entities:
            max_num_entities = len(doc_to_entities[annot["docid"]])
        # Add holder target key
        ht_key = "(" + annot["holder"] + ", " + annot["target"] + ")"
        if ht_key not in doc_to_ht_to_annot[annot["docid"]]:
            doc_to_ht_to_annot[annot["docid"]][ht_key] = annot
            if annot["label"] != 1:
                doc_to_ht_pairs_sentiment[annot["docid"]].append((annot["holder"], annot["target"]))
            if annot["label"] == 2:
                doc_to_ht_pairs_positive[annot["docid"]].append((annot["holder"], annot["target"]))
        all_data.append(annot)

print("max entities = " + str(max_num_entities))

# Sorting
all_data.sort(key=lambda x: len(x["token"]))
all_data = all_data[56:]  # just for this one
for doc in doc_to_entities:
    print(doc_to_entities[doc])
    print(doc_to_entity_inds[doc])
    zipped = zip(doc_to_entities[doc], doc_to_entity_inds[doc])
    zipped = sorted(zipped, key=lambda x: len(x[1]), reverse=True)
    doc_to_entities[doc], doc_to_entity_inds[doc] = zip(*zipped)
    print(doc_to_entities[doc])
    print(doc_to_entity_inds[doc])

# all_data = all_data[:1]
docs_chosen = []
text_chosen = []

i = 0
j = 0
for line in all_data:
    if i > 0:
        break
    doc = line["docid"]
    for ht_key in doc_to_ht_to_annot[doc]:
        j += 1
        annot = doc_to_ht_to_annot[doc][ht_key]
    for entity_index in doc_to_entity_inds[doc]:
        print(convert_to_str(annot["token"], holder_inds=entity_index))
        print()
    text_chosen.append(convert_to_str(annot["token"], entity_inds=doc_to_entity_inds[doc]))
    print("number ht pairs = " + str(j))
    i += 1
    docs_chosen.append(doc)

print(docs_chosen[0])

# Save text
for i in range(len(docs_chosen)):
    docid = docs_chosen[i]
    text = text_chosen[i]
    with open("./demo/" + docid + ".txt", "w", encoding="latin1") as wf:
        wf.write(text)

# Graphing
nodes_list = doc_to_entities[docs_chosen[0]]  # ['Israel', 'State', 'Clinton', 'US', "AIPAC", "Obama"]

edges_list = doc_to_ht_pairs_sentiment[docs_chosen[0]]
''' [("US", "Israel"),
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
("Clinton", "Israel")] '''

# Specify the edges you want here
green_edges = doc_to_ht_pairs_positive[docs_chosen[0]]
'''[("US", "Israel"),
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
   ("Clinton", "Israel")] '''

G = nx.DiGraph()
G.add_nodes_from(nodes_list)
G.add_edges_from(edges_list)

nodes_pos = graphviz_layout(G)

# values = [val_map.get(node, 0.25) for node in G.nodes()]
# print(values)

red_edges = [edge for edge in G.edges() if edge not in green_edges]
# unconnected_nodes = [node for node in G.nodes() if node not in nodes_list]

print(nodes_list)
print(edges_list)
print(green_edges)
print(red_edges)

# Need to create a layout when doing
# separate calls to draw nodes and edges
nx.draw_networkx_nodes(G, nodes_pos, nodelist=nodes_list, node_color=COLORS[:len(nodes_list)],
                       cmap=plt.get_cmap('jet'), node_size=500)
nx.draw_networkx_labels(G, nodes_pos)
nx.draw_networkx_edges(G, nodes_pos, arrowsize=50, edgelist=green_edges, edge_color='g', arrows=True)
nx.draw_networkx_edges(G, nodes_pos, arrowsize=50, edgelist=red_edges, edge_color='r', arrows=True)

# Save and display graph
plt.tight_layout()
plt.axis('auto')
plt.xlim((plt.xlim()[0] - 50, plt.xlim()[1] + 50))
plt.ylim((plt.ylim()[0] - 5, plt.ylim()[1] + 5))
plt.axis('off')
plt.savefig("./demo/" + docs_chosen[0] + ".png", bbox_inches='tight')  # save figure
plt.show()
