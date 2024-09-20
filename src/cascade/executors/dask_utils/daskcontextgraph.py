import asyncio
import socket
import time

import psutil
from dask.distributed import get_worker
from distributed.comm import connect

from cascade.contextgraph import ContextGraph


class NetworkBenchmark:
    @staticmethod
    async def benchmark_send(target, size):
        comm = await connect(target)
        start = time.time()
        await comm.write({"op": "on_benchmark", "data": b"x" * size})
        comm.close()
        return start

    @staticmethod
    def register_handler(size):
        worker = get_worker()
        worker._benchmark_event = asyncio.Event()

        def on_benchmark(data):
            assert len(data) == size
            worker._benchmark_event.set()

        worker.handlers["on_benchmark"] = on_benchmark

    @staticmethod
    async def benchmark_recv():
        worker = get_worker()
        await worker._benchmark_event.wait()
        end = time.time()
        worker.handlers.pop("on_benchmark")
        del worker._benchmark_event
        return end

    @staticmethod
    def run(dask_client, sender, receiver, size):
        reg = dask_client.submit(
            NetworkBenchmark.register_handler,
            size,
            workers=[receiver],
            pure=False,
        )
        reg.result()

        r = dask_client.submit(
            NetworkBenchmark.benchmark_recv,
            workers=[receiver],
            pure=False,
        )
        s = dask_client.submit(
            NetworkBenchmark.benchmark_send,
            receiver,
            size,
            workers=[sender],
            pure=False,
        )

        start = s.result()
        end = r.result()

        bandwidth = size / (end - start) / 1024 / 1024  # MiB/s

        return bandwidth


def fetch_worker_info(dask_worker):
    # We can also do a communication here to benchmark to each other process perhaps
    return {
        "name": dask_worker.name,
        "memory_total": dask_worker.memory_manager.memory_limit / (1024**2),  # MiB
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_speed": psutil.cpu_freq().current if psutil.cpu_freq() else 0,  # MHz
        "hostname": socket.gethostname(),
        "uri": dask_worker.address,
    }


def create_dask_context_graph(client):
    # Fetch worker information across the Dask cluster
    worker_info = client.run(fetch_worker_info)
    context_graph = ContextGraph()

    print("Worker info")
    print(worker_info)

    # Build nodes for each worker using worker names
    for info in worker_info.values():
        context_graph.add_node(
            info["name"], "cpu", info["cpu_speed"], info["memory_total"], info["uri"]
        )

    # Group workers by hostname
    host_groups = {}
    for info in worker_info.values():
        host_groups.setdefault(info["hostname"], []).append(info)

    # Connect workers within the same host
    for workers in host_groups.values():
        for i in range(len(workers)):
            for j in range(i + 1, len(workers)):
                bandwidth = NetworkBenchmark.run(
                    client, workers[i]["uri"], workers[j]["uri"], 10 * 1024 * 1024
                )
                print(
                    f"Bandwidth: {bandwidth} MiB/s between worker {workers[i]['name']} and worker {workers[j]['name']}"
                )
                context_graph.add_edge(
                    workers[i]["name"],
                    workers[j]["name"],
                    bandwidth=bandwidth,
                    latency=0,
                )

    # And across different hosts
    hosts = list(host_groups.keys())
    for i in range(len(hosts)):
        host1 = hosts[i]
        workers1 = host_groups[host1]
        for j in range(i + 1, len(hosts)):
            host2 = hosts[j]
            workers2 = host_groups[host2]
            for worker1 in workers1:
                for worker2 in workers2:
                    bandwidth = NetworkBenchmark.run(
                        client,
                        worker1["uri"],
                        worker2["uri"],
                        10 * 1024 * 1024,
                    )
                    print(
                        f"Bandwidth: {bandwidth} MiB/s between worker {worker1['name']}"
                        + f" and worker {worker2['name']}"
                    )
                    context_graph.add_edge(
                        worker1["name"],
                        worker2["name"],
                        bandwidth=bandwidth,
                        latency=0,
                    )

    print(context_graph)
    return context_graph
