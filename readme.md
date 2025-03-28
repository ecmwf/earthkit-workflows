[![ci](https://github.com/ecmwf/cascade/actions/workflows/ci.yml/badge.svg)](https://github.com/ecmwf/cascade/actions/workflows/ci.yml)

![Image](cascade.png)

> **DISCLAIMER**
> This project is **BETA** and will be **experimental** for the foreseeable future. Interfaces and functionality are likely to change, and the project itself may be scrapped. **DO NOT** use this software in any project/software that is operational.

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

## Development

Install via `pip` with:

```
$ pip install -e cascade
```

Install precommit hooks
```
$ pip install pre-commit
$ pre-commit install
```

## Executions
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

## License

```
Copyright 2022, European Centre for Medium Range Weather Forecasts.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
```
