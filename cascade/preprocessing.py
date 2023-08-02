from typing import Tuple, List
import copy
import networkx as nx
import numpy as np

from .graphs import Task, Communication, TaskGraph


def dedensify(graph: TaskGraph, threshold: int) -> Tuple[TaskGraph, List[Task]]:
    """
    Wraps networkx.dedensify, which is an algorithm for reducing the degree of nodes with
    in_degree > threshold by compressing nodes and introducing new edges.

    Returns dedensified graph and the new nodes resulting from compression existing nodes
    """
    dedensified_graph, compressed_nodes = nx.dedensify(graph, threshold, copy=True)
    dedensified_graph.node_dict = copy.deepcopy(graph.node_dict)
    print(compressed_nodes)

    relabellings = {}
    for index, compressed_node in enumerate(compressed_nodes):
        # TODO: Deduce operation based on operation of compressed nodes e.g.
        # min, max, sum along the same axis
        memory = 0
        cost = 0
        in_comm_size = 0
        for in_node, _ in dedensified_graph.in_edges(compressed_node):
            comm_size = np.max(
                [
                    graph[in_node][out_node]["obj"].size
                    for _, out_node in dedensified_graph.out_edges(compressed_node)
                ]
            )
            dedensified_graph[in_node][compressed_node]["obj"] = Communication(
                in_node, compressed_node, comm_size
            )
            memory += in_node.out_memory
            in_comm_size += comm_size

        for _, out_node in dedensified_graph.out_edges(compressed_node):
            dedensified_graph[compressed_node][out_node]["obj"] = Communication(
                compressed_node, out_node, in_comm_size
            )
            cost = max(cost, out_node.cost)

        # Compressed nodes are simply a concatenation of inputs so out_memory same as in_memory
        new_node = Task(cost, memory, memory, name=f"compressed{index}")
        relabellings[compressed_node] = new_node
        dedensified_graph.node_dict[new_node.name] = new_node

    nx.relabel_nodes(dedensified_graph, relabellings, copy=False)

    return dedensified_graph, list(relabellings.values())


def chunk(graph: TaskGraph, memory_constraint: int) -> nx.DiGraph:
    """
    Chunks the computation of a node if the memory requirement of the node exceeds
    memory_constraint by dividing inputs between new intermediate tasks for partial computation.
    Returns a copy of the input graph with chunking applied.
    """
    chunked_graph = copy.deepcopy(graph)
    for node in nx.lexicographical_topological_sort(chunked_graph):
        # For now, assume all node operations can be chunked.
        # TODO:
        # - raise error in the case when operation can not be chunked and memory exceeds constraint
        # - implement a load balancing algorithm for assigning tasks to chunks
        if node.memory > memory_constraint:
            print(f"Chunking node {node} memory {node.memory}")

            in_edges = list(chunked_graph.in_edges(node, data="obj"))
            i_chunk = 0
            memory = 0
            comm_size = 0
            chunk_in_nodes = []
            for edge_index in range(len(in_edges) + 1):
                if edge_index < len(in_edges):
                    next_node, _, comm = in_edges[edge_index]
                    assert (
                        next_node.memory <= memory_constraint
                    ), f"Found in edge with node memory {next_node.memory} exceeding constraint"

                if (
                    edge_index == len(in_edges)
                ) or memory + next_node.out_memory > memory_constraint:
                    new_task_name = f"{node.name}_chunk{i_chunk}"
                    assert (
                        len(chunk_in_nodes) > 0
                    ), f"No in edges for chunk {new_task_name}"
                    assert (
                        memory <= memory_constraint
                    ), f"Created chunk {new_task_name} with memory {memory} above constraint {memory_constraint}"
                    # TODO: New out_memory should be deduced from node operation on inputs
                    chunked_graph.add_task(node.cost, in_memory=memory, out_memory=node.out_memory, name=new_task_name)
                    print(f"Adding new task {new_task_name}")
                    for x, size in chunk_in_nodes:
                        chunked_graph.add_comm_edge(x.name, new_task_name, size)
                    chunked_graph.add_comm_edge(new_task_name, node.name, comm_size)

                    if edge_index == len(in_edges):
                        break
                    memory = 0
                    comm_size = 0
                    chunk_in_nodes = []
                    i_chunk += 1

                memory += next_node.out_memory
                comm_size += comm.size
                chunk_in_nodes.append((next_node, comm.size))

            chunked_graph.remove_edges_from(in_edges)

            # Recompute node memory from new chunked tasks
            node.in_memory = sum([x.out_memory for x, _ in chunked_graph.in_edges(node)])
            print(f"Node {node} post-chunking memory {node.memory}")

    return chunked_graph
