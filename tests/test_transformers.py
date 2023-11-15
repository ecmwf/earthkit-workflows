import os
import pytest

from cascade.cascade import Cascade
from cascade.transformers import to_dask_graph

from helpers.mock import MockArgs

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    "product, config",
    [
        ["prob", f"{ROOT_DIR}/templates/prob.yaml"],
        ["anomaly_prob", f"{ROOT_DIR}/templates/t850.yaml"],
        ["wind", f"{ROOT_DIR}/templates/wind.yaml"],
        ["ensms", f"{ROOT_DIR}/templates/ensms.yaml"],
        ["extreme", f"{ROOT_DIR}/templates/extreme.yaml"],
    ],
)
def test_dask_transform(product, config):
    graph = Cascade.graph(product, MockArgs(config))

    dask_graph = to_dask_graph(graph)
    assert all([isinstance(x, tuple) for x in dask_graph.items()])
