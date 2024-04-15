import pytest

from cascade.contextgraph import ContextGraph
from cascade.graph import Graph
from cascade.taskgraph import Task


@pytest.fixture
def context():
    ret = ContextGraph()
    ret.add_node("gpu_1", type="GPU", speed=10, memory=40)
    ret.add_node("gpu_2", type="GPU", speed=10, memory=20)
    ret.add_node("gpu_3", type="GPU", speed=5, memory=40)
    ret.add_node("gpu_4", type="GPU", speed=5, memory=20)
    ret.add_edge("gpu_1", "gpu_2", bandwidth=0.1, latency=1)
    ret.add_edge("gpu_1", "gpu_3", bandwidth=0.02, latency=3)
    ret.add_edge("gpu_1", "gpu_4", bandwidth=0.02, latency=3)
    ret.add_edge("gpu_2", "gpu_3", bandwidth=0.02, latency=3)
    ret.add_edge("gpu_2", "gpu_4", bandwidth=0.02, latency=3)
    ret.add_edge("gpu_3", "gpu_4", bandwidth=0.1, latency=1)
    return ret


def example_graph(num_inputs: int):
    mean = Task(name="mean", payload="mean")
    for index in range(num_inputs):
        read = Task(name=f"read-{index}", payload=f"read-{index}")
        sh2gp = Task(name=f"sh2gp-{index}", payload=f"sh2gp-{index}")
        sh2gp.inputs[f"inputs0"] = read.get_output()
        mean.inputs[f"inputs{index}"] = sh2gp.get_output()
    return Graph([mean])
