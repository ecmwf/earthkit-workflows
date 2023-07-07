import networkx as nx
import flask
import json
import random

G = nx.gnp_random_graph(100,0.1,directed=True)

DAG = nx.DiGraph([(u,v,{'weight':random.randint(-10,10)}) for (u,v) in G.edges() if u<v])

print(nx.is_directed_acyclic_graph(DAG))

print(nx.dag_longest_path(DAG))


def visualize(graph):
    d = nx.readwrite.json_graph.node_link_data(graph)
    json.dump(d, open('force/force.json','w'))
    print('Wrote node-link JSON data to force/force.json')

    app = flask.Flask(__name__, static_folder="force")

    @app.route('/<path:path>')
    def static_proxy(path):
        return app.send_static_file(path)
    print('\nGo to http://localhost:8000/force.html to see the example\n')
    app.run(port=8000)

