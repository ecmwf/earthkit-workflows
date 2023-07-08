import networkx as nx
import matplotlib.pyplot as plt
import copy


G = nx.DiGraph()

# Create 10 output nodes
for i in range(10):
    G.add_node(f"output_{i}", GPU={"cost": 10, "memory": 1})


# Create 50 reader nodes, each with an additional processing node
for i in range(50):
    G.add_node(f"read_{i}", GPU={"cost": 10, "memory": 1})
    G.add_node(f"sh2gp_{i}", GPU={"cost": 10, "memory": 1})
    G.add_edge(f"read_{i}", f"sh2gp_{i}", size=0.5)

# Create processes which require several of the inputs to produce the outputs
for i in range(10):
    G.add_node(f"mean_{i}", GPU={"cost": 3, "memory": 4})
    for j in range(7):
        G.add_edge(f"sh2gp_{min(i*5+j,49)}", f"mean_{i}", size=1)
    G.add_edge(f"mean_{i}", f"output_{i}", size=1)

# Create a process which requires all of the means to produce the output
G.add_node("mean_all", GPU={"cost": 3, "memory": 4})
for i in range(10):
    G.add_edge(f"mean_{i}", "mean_all", size=1)
G.add_edge("mean_all", "output_all", size=1)

# nx.write_dot(G,'test.dot')
# pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args='-Nshape=box')
# nx.draw_networkx(G, pos=pos, with_labels=False)
# plt.show()


contexts = nx.Graph()
contexts.add_node("gpu_1", type="GPU", speed=10, memory=10)
contexts.add_node("gpu_2", type="GPU", speed=10, memory=10)
contexts.add_node("gpu_3", type="GPU", speed=5, memory=10)
contexts.add_node("gpu_4", type="GPU", speed=5, memory=10)
contexts.add_edge("gpu_1", "gpu_2", bandwidth=100)
contexts.add_edge("gpu_3", "gpu_4", bandwidth=100)
contexts.add_edge("gpu_1", "gpu_3", bandwidth=20)
contexts.add_edge("gpu_2", "gpu_4", bandwidth=20)


class ExecutionOptimizer:
    def __init__(self, task_graph, context_graph):
        self.task_graph = copy.deepcopy(task_graph)
        self.context_graph = copy.deepcopy(context_graph)

    def first_guess(self):
        assert self.task_graph.is_directed()

        # Get all contexts
        contexts = self.context_graph.nodes.data()
        for _, context in contexts:
            context["next_available_time"] = 0
            context["task_list"] = []
        
        for _, task in self.task_graph.nodes.data():
            task["finished"] = False

        # Get all the nodes which have no incoming edges
        eligible = [n for n, d in self.task_graph.in_degree() if d == 0]

        while len(eligible) > 0:

            # Find the context with the earliest available time
            context = min(contexts, key=lambda x: x[1]["next_available_time"])

            if len(context[1]["task_list"]) > 0:
                task = context[1]["task_list"][-1]
                self.task_graph.nodes[task]["finished"] = True
                for successor in self.task_graph.successors(task):
                    if all([self.task_graph.nodes[predecessor]["finished"] for predecessor in self.task_graph.predecessors(successor)]):
                        eligible.append(successor)

            # Find the eligible node with the highest cost and most dependents
            eligible.sort(key=lambda x: (self.task_graph.nodes[x]["GPU"]["cost"], self.task_graph.out_degree[x]))

            # Assign the node to the context
            node = eligible.pop()
            context[1]["task_list"].append(node)
            context[1]["next_available_time"] += self.task_graph.nodes[node]["GPU"]["cost"]/context[1]["speed"]

        for context in contexts:
            print(f"{context[0]} has total execution time of {context[1]['next_available_time']}")
            print(context[1]["task_list"])
    

ExecutionOptimizer(G, contexts).first_guess()

            



