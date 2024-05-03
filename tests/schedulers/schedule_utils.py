from cascade.graph import Graph
from cascade.taskgraph import Task


def example_graph(num_inputs: int):
    mean = Task(name="mean", payload="mean")
    for index in range(num_inputs):
        read = Task(name=f"read-{index}", payload=f"read-{index}")
        sh2gp = Task(name=f"sh2gp-{index}", payload=f"sh2gp-{index}")
        sh2gp.inputs["inputs0"] = read.get_output()
        mean.inputs[f"inputs{index}"] = sh2gp.get_output()
    return Graph([mean])
