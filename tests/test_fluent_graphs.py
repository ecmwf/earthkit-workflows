import pytest
import os

from cascade.graph_config import Config
from cascade.cascade import Cascade

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
def test_graph_construction(product, config):
    cfg = Config(config)
    Cascade.graph(product, cfg)
