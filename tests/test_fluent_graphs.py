import pytest
import os

from ppgraph import pyvis
from cascade.cascade import Cascade

from helpers.mock import mock_args, mock_cluster_args

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def node_info_ext(sinks, node):
    info = pyvis.node_info(node)
    info["color"] = "#648FFF"
    if not node.inputs:
        info["shape"] = "diamond"
        info["color"] = "#DC267F"
    elif node in sinks:
        info["shape"] = "triangle"
        info["color"] = "#FFB000"
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
        ["wind", f"{ROOT_DIR}/templates/wind.yaml", 168],
        ["ensms", f"{ROOT_DIR}/templates/ensms.yaml", 180],
        ["extreme", f"{ROOT_DIR}/templates/extreme.yaml", 150],
        ["quantiles", f"{ROOT_DIR}/templates/quantiles.yaml", 92],
    ],
)
def test_graph_construction(product, config, expected_num_nodes):
    graph = Cascade.graph(product, mock_args(config))
    assert len([x for x in graph.nodes()]) == expected_num_nodes


def test_cluster_graph():
    # With spread compute
    mock_args = mock_cluster_args(f"{ROOT_DIR}/templates/clustereps.yaml")
    graph = Cascade.graph("clustereps", mock_args)
    assert len([x for x in graph.nodes()]) == 64

    # With spread
    mock_args.spread = "fileset:spread_z500"
    graph = Cascade.graph("clustereps", mock_args)
    assert len([x for x in graph.nodes()]) == 33


def test_unregistered():
    with pytest.raises(Exception):
        Cascade.graph("test")
