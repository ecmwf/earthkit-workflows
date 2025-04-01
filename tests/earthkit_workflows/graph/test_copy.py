from earthkit.workflows.graph import copy_graph
from earthkit.workflows.graph.samplegraphs import multi


def test_copy():
    g = multi()
    gc = copy_graph(g)
    assert g == gc
    for node in gc.nodes():
        orig = g.get_node(node.name)
        assert orig.name == node.name
        assert orig.outputs == node.outputs
        assert sorted(orig.inputs.keys()) == sorted(node.inputs.keys())
        assert orig is not node
