import asyncio
import socket
import time

import psutil
from dask.distributed import get_worker

# from dask.distributed import connect
from distributed.comm import connect

from cascade.contextgraph import ContextGraph


class NetworkBenchmark:

    async def benchmark_send(target, size):
        comm = await connect(target)
        # print(comm.same_host)
        start = time.time()
        await comm.write({"op": "on_benchmark", "data": b"x" * size})
        comm.close()
        return start

    async def benchmark_recv(source, size):

        event = asyncio.Event()

        def on_benchmark(data):
            assert len(data) == size
            event.set()

        worker = get_worker()
        worker.handlers["on_benchmark"] = on_benchmark
        await event.wait()
        worker.handlers.pop("on_benchmark")
        end = time.time()
        return end

    def run(dask_client, sender, receiver, size):

        r = dask_client.submit(
            NetworkBenchmark.benchmark_recv,
            sender,
            size,
            workers=[receiver],
            pure=False,
        )
        # time.sleep(0.01)
        # I don't think this is safe, if the sender runs before the receiver then the
        # handler won't be in place, but I'm not sure.
        s = dask_client.submit(
            NetworkBenchmark.benchmark_send,
            receiver,
            size,
            workers=[sender],
            pure=False,
        )
        # time.sleep(0.01)

        start = s.result()
        end = r.result()

        bandwidth = size / (end - start) / 1024 / 1024  # MiB/s

        return bandwidth


def fetch_worker_info(dask_worker):

    # We can also do a communication here to benchmark to each other process perhaps
    return {
        "name": dask_worker.name,
        "memory_total": dask_worker.memory_manager.memory_limit / (1024**3),  # GB
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
    for worker, info in worker_info.items():
        host_groups.setdefault(info["hostname"], []).append(info)

    # Connect workers within the same host
    for host, workers in host_groups.items():
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
    for host1, workers1 in host_groups.items():
        for host2, workers2 in host_groups.items():
            if host1 != host2:
                for worker1 in workers1:
                    for worker2 in workers2:
                        bandwidth = NetworkBenchmark.run(
                            client,
                            workers[i]["uri"],
                            workers[j]["uri"],
                            10 * 1024 * 1024,
                        )
                        print(
                            f"Bandwidth: {bandwidth} MiB/s between worker {workers[i]['name']}"
                            + f"and worker {workers[j]['name']}"
                        )
                        context_graph.add_edge(
                            workers[i]["name"],
                            workers[j]["name"],
                            bandwidth=bandwidth,
                            latency=0,
                        )

    # for sender in worker_info.keys():
    #     for receiver in worker_info.keys():
    #         if sender != receiver:
    #             NetworkBenchmark.run(client, sender, receiver, 1000)

    print(context_graph)

    # context_graph.visualise("contextgraph.html")

    return context_graph
