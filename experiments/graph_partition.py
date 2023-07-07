import networkx as nx


G = nx.DiGraph()


# Create 10 output nodes
for i in range(10):
    G.add_node(f"output_{i}")


# Create 50 reader nodes, each with an additional processing node
for i in range(50):
    G.add_node(f"read_{i}")
    G.add_node(f"sh2gp_{i}")
    G.add_edge(f"read_{i}", f"sh2gp_{i}")

# Create processes which require several of the inputs to produce the outputs
for i in range(10):
    G.add_node(f"mean_{i}")
    for j in range(7):
        G.add_edge(f"sh2gp_{min(i*5+j,49)}", f"mean_{i}")
    G.add_edge(f"mean_{i}", f"output_{i}")

# Create a process which requires all of the means to produce the output
G.add_node("mean_all")
for i in range(10):
    G.add_edge(f"mean_{i}", "mean_all")
G.add_edge("mean_all", "output_all")

# nx.write_dot(G,'test.dot')
pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args='-Nshape=box')
import matplotlib.pyplot as plt
nx.draw_networkx(G, pos=pos, with_labels=False)
plt.show()