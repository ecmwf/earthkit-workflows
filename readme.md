<p align="center">
  <picture>
    <source srcset="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-workflows-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-workflows-light.svg" height="120">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/production_chain_badge.svg" alt="ECMWF Software EnginE">
  </a>
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/emerging_badge.svg" alt="Maturity Level">
  </a>
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
  <a href="https://github.com/ecmwf/earthkit-workflows/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-workflows?color=purple&label=Release" alt="Latest Release">
  </a>
</p>

<p align="center">
  <a href="#installation">Installation</a>
  •
  <a href="#quick-start">Quick Start</a>
  •
  <a href="#documentation">Documentation</a>
</p>

> \[!IMPORTANT\]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

Earthkit Workflows is a Python library for declaring earthkit task as DAGs.
It contains an internal `cascade` engine for scheduling and executing task graphs almost optimally across heterogeneous platforms with complex network technologies and topologies.
It effectively performs task-based parallelism across CPUs, GPUs, distributed systems (HPC), and any combination thereof.
It is designed for a no-IO approach, where expensive storage of intermediate data is minimised whilst maximising all available transport technologies between different hardware.

Cascade is designed to work on well-profiled task graphs, where:
* the task graph is a static DAG,
* the DAG nodes are defined by tasks with well-known execution times,
* the DAG edges are defined by data dependencies with well-known data sizes,
* the characteristics of the hardware (processors, network connections) are known.

Earthkit Workflows allows for declaring such task graphs using a neat fluent API, and interoperates pleasantly with the rest of the Earthkit ecosystem.

## Installation

Install via `pip` with:

```
$ pip install 'earthkit-workflows[all]'
```

For development, you can use `pip install -e .` though there is currently an issue with earthkit masking. Additionally you may want to install pre-commit hooks via
```
$ pip install pre-commit
$ pre-commit install
```

## Quick Start

*Note*: this section is moderately outdated.

We support two regimes for cascade executions -- local mode (ideal for developing and debugging small graphs) and distributed mode (assumed for slurm & HPC).

To launch in local mode, in your python repl / jupyno:
```
import cascade.benchmarks.job1 as j1
import cascade.benchmarks.distributed as di
import cloudpickle

spec = di.ZmqClusterSpec.local(j1.get_prob())
print(spec.controller.outputs)
# prints out:
# {DatasetId(task='mean:dc9d90 ...
# defaults to all "sinks", but can be overridden

rv = di.launch_from_specs(spec, None)

for key, value in rv.outputs.items():
    deser = cloudpickle.loads(value)
    print(f"output {key} is of type {type(deser)}")
```

For distributed mode, launch
```
./scripts/launch_slurm.sh ./localConfigs/<your_config.sh>
```
Inside the `<your_config.sh>`, you define size of the cluster, logging directory output, which job to run... Pay special attention to definitions of your `venv` and `LD_LIBRARY_PATH` etc -- this is not autotamed.

Both of these examples hardcode particular job, `"job1"`, which is a benchmarking thing.
Most likely, you want to define your own -- for the local mode, just pass `cascade.Graph` instance to the call; in the dist mode, you need to provide that instance in the `cascade.benchmarks.__main__` modules instead (ideally by extending the `get_job` function).

There is also `python -m cascade.benchmarks local <..>` -- you may use that as an alternative path to local mode, for your own e2e tests.

## Documentation

Not yet available.

## License

See [license](./LICENSE).
