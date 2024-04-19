import psutil
import socket

from cascade.contextgraph import ContextGraph

def fetch_worker_info(dask_worker):

    # We can also do a communication here to benchmark to each other process perhaps
    return {
        'name': dask_worker.name,
        'memory_total': psutil.virtual_memory().total / (1024 ** 3),  # GB
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_speed': psutil.cpu_freq().current if psutil.cpu_freq() else 0,  # MHz
        'hostname' : socket.gethostname()
    }


def create_dask_context_graph(client):

    # Fetch worker information across the Dask cluster
    worker_info = client.run(fetch_worker_info)
    context_graph = ContextGraph()

    print("Worker info")
    print(worker_info)

    # Build nodes for each worker using worker names
    for info in worker_info.values():
        context_graph.add_node(info['name'], 'cpu', info['cpu_speed'], info['memory_total'])

    # Group workers by hostname
    host_groups = {}
    for worker, info in worker_info.items():
        host_groups.setdefault(info['hostname'], []).append(info['name'])

    # Connect workers within the same host
    for host, workers in host_groups.items():
        for i in range(len(workers)):
            for j in range(i + 1, len(workers)):
                context_graph.add_edge(workers[i], workers[j], bandwidth=5, latency=0)

    # And across different hosts
    for host1, workers1 in host_groups.items():
        for host2, workers2 in host_groups.items():
            if host1 != host2:
                for worker1 in workers1:
                    for worker2 in workers2:
                        context_graph.add_edge(worker1, worker2, bandwidth=1, latency=0.05)

    print("Context graph")
    print(context_graph)

    # context_graph.visualise("contextgraph.html")

    return context_graph