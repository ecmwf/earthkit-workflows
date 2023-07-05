# Cascade

Cascade is designed to execute task graphs optimally across heterogeneous platforms with complex network technologies and topologies. It effectively performs task-based parallelism across CPUs, GPUs, distributed systems (HPC), and any combination thereof. It is designed for a no-IO approach, where expensive storage of intermediate data is minimised whilst maximising all available transport technologies between different hardware, via UCX. Cascade delivers this power through a simple Python API.


Cascade is designed to work on well-profiled task graphs, where:
* the DAG is static
* the DAG edges are defined by data dependencies
* the tasks have a well-known execution time
* the size of each task output is known in advance

There are three components to Cascade:

* the graph partioner
* the graph executor
* the transport abstraction layer