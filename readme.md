![Image](cascade.png)

Cascade is designed to execute task graphs optimally across heterogeneous platforms with complex network technologies and topologies. It effectively performs task-based parallelism across CPUs, GPUs, distributed systems (HPC), and any combination thereof. It is designed for a no-IO approach, where expensive storage of intermediate data is minimised whilst maximising all available transport technologies between different hardware.

Cascade is designed to work on well-profiled task graphs, where:
* the task graph is a static DAG
* the DAG nodes are defined by tasks with well-known execution times
* the DAG edges are defined by data dependencies with well-known data sizes
* the characteristics of the hardware (processors, network connections) are known

There are two main components to Cascade:

## The Executor

The executor is responsible for executing the plan created by the scheduler on the target hardware. Each executor should be capable of simulating its execution plan.

## The Scheduler

The scheduler is responsible for creating an execution plan for the DAG. For schedulers based on optimization, the scheduler is likely to use a simulated execution as a fitness function for the optimization.

# Quick Start

```
 python3 ./tests/test_depth_first_scheduler.py
```
