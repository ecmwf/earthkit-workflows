from dask.distributed import Client, LocalCluster
import psutil

def fetch_worker_info(dask_worker):

    # We can also do a communication here to benchmark to each other process perhaps
    return {
        'name': dask_worker.name,
        'memory_total': psutil.virtual_memory().total / (1024 ** 3),  # GB
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_speed': psutil.cpu_freq().current if psutil.cpu_freq() else 0  # MHz
    }

def create_dask_context_graph(client):

    # Runs on each worker, outside the scheduler
    worker_info = client.run(fetch_worker_info)
    context_graph = ContextGraph()

    for worker, info in worker_info.items():
        context_graph.add_node(worker, **info)

    workers = list(worker_info.keys())
    for i in range(len(workers) - 1):
        context_graph.add_edge(workers[i], workers[i+1], bandwidth=1.0, latency=5.0)  # Example values

    return context_graph

if __name__ == '__main__':
    cluster = LocalCluster()
    client = Client(cluster)

    system_context_graph = create_dask_context_graph(client)
    system_context_graph.display()

    client.close()
    cluster.close()