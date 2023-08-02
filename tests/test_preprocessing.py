from typing import List, Tuple
import numpy as np
import networkx as nx
import pytest

from cascade.graphs import TaskGraph, Task
from cascade.preprocessing import dedensify, chunk


def sample_graph(
    num_steps: int, num_ensembles: int, windows: List[Tuple[int, int]]
) -> TaskGraph:
    graph = TaskGraph()

    for i in range(num_steps):
        graph.add_task(cost=100, in_memory=1, out_memory=0, name=f"output_step{i}")

    for ens in range(num_ensembles):
        for step in range(num_steps):
            graph.add_task(cost=100, in_memory=0, out_memory=1, name=f"read_ens{ens}_step{step}")
            graph.add_task(cost=100, in_memory=1, out_memory=1, name=f"sh2gp_ens{ens}_step{step}")
            graph.add_comm_edge(
                f"read_ens{ens}_step{step}", f"sh2gp_ens{ens}_step{step}", size=0.5
            )

    for step in range(num_steps):
        graph.add_task(cost=30, in_memory=num_ensembles, out_memory=1, name=f"mean_step{step}")
        for ens in range(num_ensembles):
            graph.add_comm_edge(
                f"sh2gp_ens{ens}_step{step}", f"mean_step{step}", size=1
            )
        graph.add_comm_edge(f"mean_step{step}", f"output_step{step}", size=1)

    for window_start, window_end in windows:
        window_length = window_end - window_start + 1
        graph.add_task(
            cost=30,
            in_memory=num_ensembles*window_length,
            out_memory=1,
            name=f"mean_step{window_start}-{window_end}"
        )
        for ens in range(num_ensembles):
            for step in range(window_start, window_end + 1):
                graph.add_comm_edge(
                    f"sh2gp_ens{ens}_step{step}",
                    f"mean_step{window_start}-{window_end}",
                    size=1,
                )     
        graph.add_task(
            cost=30, in_memory=1, out_memory=0, name=f"output_step{window_start}-{window_end}"
        )
        graph.add_comm_edge(
            f"mean_step{window_start}-{window_end}",
            f"output_step{window_start}-{window_end}",
            size=1,
        )

    return graph


def test_dedensify():
    num_ensembles = 5
    num_steps = 5
    windows = [(0, 3), (3, 4)]
    graph = sample_graph(num_steps, num_ensembles, windows)

    dedensified_graph, compressed_nodes = dedensify(graph, 3)
    assert np.all([isinstance(x, Task) for x in dedensified_graph])
    assert np.all([comm.size > 0 for _, _, comm in dedensified_graph.edges(data="obj")])
    assert len(compressed_nodes) > 0


@pytest.mark.parametrize("num_ens, num_steps, windows, mem_satisfied", 
                          [
                              [5, 5, [(0, 3), (3, 4)], False],
                              [1, 4, [(0, 3)], True]
                          ])
def test_chunk(num_ens, num_steps, windows, mem_satisfied):
    graph = sample_graph(num_steps, num_ens, windows)

    mem_constraint = 2
    chunked_graph = chunk(graph, mem_constraint)
    # Chunking should not create disconnected components
    assert nx.is_weakly_connected(chunked_graph)
    assert np.all([x.memory <= mem_constraint for x in chunked_graph]) == mem_satisfied
    assert np.all([isinstance(x, Task) for x in chunked_graph])
    assert np.all([comm.size > 0 for _, _, comm in chunked_graph.edges(data="obj")])
