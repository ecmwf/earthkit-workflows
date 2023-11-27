import os
import random

from ppgraph import Node, Sink, Graph, Transformer
from cascade.cascade import Cascade
from cascade.graphs import Task, TaskGraph, ContextGraph
from cascade.scheduler import AnnealingScheduler

from helpers.mock import mock_args


ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def setup_context():
    context = ContextGraph()
    context.add_node("gpu_1", type="GPU", speed=10, memory=40)
    context.add_node("gpu_2", type="GPU", speed=10, memory=20)
    context.add_node("gpu_3", type="GPU", speed=5, memory=40)
    context.add_node("gpu_4", type="GPU", speed=5, memory=20)
    context.add_edge("gpu_1", "gpu_2", bandwidth=0.1, latency=1)
    context.add_edge("gpu_1", "gpu_3", bandwidth=0.02, latency=3)
    context.add_edge("gpu_1", "gpu_4", bandwidth=0.02, latency=3)
    context.add_edge("gpu_2", "gpu_3", bandwidth=0.02, latency=3)
    context.add_edge("gpu_2", "gpu_4", bandwidth=0.02, latency=3)
    context.add_edge("gpu_3", "gpu_4", bandwidth=0.1, latency=1)
    return context


class _AssignRandomResources(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = Task(node.name, node.outputs.copy(), node.payload)
        newnode.inputs = inputs
        newnode.cost = random.randrange(1, 100)
        newnode.in_memory = random.randrange(1, 2)
        newnode.out_memory = random.randrange(1, 2)
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> TaskGraph:
        return TaskGraph(sinks)


def add_resources(graph: Graph) -> TaskGraph:
    return _AssignRandomResources().transform(graph)


def test_depth_first_scheduler():
    context = setup_context()
    graph = Cascade.graph("anomaly_prob", mock_args(f"{ROOT_DIR}/templates/t850.yaml"))
    schedule = Cascade.schedule(add_resources(graph), context)
    print(schedule)

    execution = Cascade.simulate(schedule)
    print(execution)


def test_annealing_scheduler():
    context = setup_context()
    graph = Cascade.graph("anomaly_prob", mock_args(f"{ROOT_DIR}/templates/t850.yaml"))
    scheduler = AnnealingScheduler(add_resources(graph), context)
    schedule = scheduler.create_schedule(num_temp_levels=10, num_tries=10)
    execution = Cascade.simulate(schedule)
    print(f"With Communications:", execution)
