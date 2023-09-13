import pytest
import os
import functools

from ppgraph import pyvis, Sink

from cascade.graph_config import Config
from cascade.cascade import Cascade

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def node_info_ext(sinks, node):
    info = pyvis.node_info(node)
    if not node.inputs:
        info["shape"] = "diamond"
        info["color"] = "red"
    elif node in sinks:
        info["shape"] = "triangle"
        info["color"] = "green"
    if node.payload is not None:
        t = []
        if "title" in info:
            t.append(info["title"])
        func, *args = node.payload
        t.append(f"Function: {func}")
        if args:
            t.append("Arguments:")
            t.extend(f"- {arg!r}" for arg in args)
        info["title"] = "\n".join(t)
    return info


@pytest.mark.parametrize(
    "product, config, expected_num_nodes",
    [
        ["prob", f"{ROOT_DIR}/templates/prob.yaml", 355],
        ["anomaly_prob", f"{ROOT_DIR}/templates/t850.yaml", 260],
        ["wind", f"{ROOT_DIR}/templates/wind.yaml", 132],
        ["ensms", f"{ROOT_DIR}/templates/ensms.yaml", 180],
        ["extreme", f"{ROOT_DIR}/templates/extreme.yaml", 150],
    ],
)
def test_graph_construction(product, config, expected_num_nodes):
    cfg = Config(config)
    graph = Cascade.graph(product, cfg)
    if True:
        pyvis_graph = pyvis.to_pyvis(
            graph,
            notebook=True,
            cdn_resources="remote",
            height="1500px",
            node_attrs=functools.partial(node_info_ext, graph.sinks),
            hierarchical_layout=False,
        )
        pyvis_graph.show(
            f"/etc/ecmwf/nfs/dh1_home_a/mawj/Documents/cascade/{product}_graph.html"
        )
    print([x for x in graph.nodes()])
    # assert len(pyvis_graph.nodes) == expected_num_nodes
    assert len([x for x in graph.nodes()]) == expected_num_nodes
