import os

from cascade.cascade import Cascade
from cascade.graphs import ContextGraph
from cascade.graph_config import Config
from cascade.scheduler import AnnealingScheduler


ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def setup_context():
    context = ContextGraph()
    context.add_node("gpu_1", type="GPU", speed=10, memory=40)
    context.add_node("gpu_2", type="GPU", speed=10, memory=20)
    context.add_node("gpu_3", type="GPU", speed=5, memory=20)
    context.add_node("gpu_4", type="GPU", speed=5, memory=20)
    context.add_edge("gpu_1", "gpu_2", bandwidth=0.1, latency=1)
    context.add_edge("gpu_1", "gpu_3", bandwidth=0.02, latency=3)
    context.add_edge("gpu_1", "gpu_4", bandwidth=0.02, latency=3)
    context.add_edge("gpu_2", "gpu_3", bandwidth=0.02, latency=3)
    context.add_edge("gpu_2", "gpu_4", bandwidth=0.02, latency=3)
    context.add_edge("gpu_3", "gpu_4", bandwidth=0.1, latency=1)
    return context


def test_depth_first_scheduler():
    context = setup_context()
    graph = Cascade.graph("anomaly_prob", Config(f"{ROOT_DIR}/templates/t850.yaml"))
    schedule = Cascade.create_schedule(graph, context)
    print(schedule)

    execution = Cascade.simulate(schedule)
    print(execution)


# def test_annealing_scheduler(self):
#     context = setup_context()
#     graph = Cascade.graph("anomaly_prob", Config(f"{ROOT_DIR}/templates/t850.yaml"))
#     schedule = AnnealingScheduler(graph, context).create_schedule()
#     execution = Cascade.simulate(schedule)
#     print(f"With Communications:", execution)
