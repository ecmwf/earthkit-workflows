# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
