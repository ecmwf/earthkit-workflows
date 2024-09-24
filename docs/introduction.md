Introduction
============

Cascade has three main components:
- **Action** defines the language for constructing task graphs and the accompanying backend objects the graph operations on
- **Schedulers** for scheduling the task graphs based on task resource requirements
- **Executors** for executing the task graph 

Fluent
------

The function cascading API for constructing graphs consists low level methods such as:
- ``reduce``
- ``join`` 
- ``broadcast``
- ``expand``
- ``transform``
- ``map``
- ``flatten``

which operate on the array of nodes in the graph, creating new nodes, and return another action object. 

The figure below shows a ``Action.reduce`` operation on a three-dimensional array of nodes over the `parameter`
dimension, which returns ``Action`` containing a two-dimensional array of nodes on the right-hand side.
<center>
<img src="reduce.png" width="400"/>
</center>

Most of the methods require a ``Callable`` which specifies the function to be applied on the array of nodes. For example, for ``Action.reduce`` the signature is 
```python
def reduce(self, payload: Callable, dim: str = "") -> "Action":
```
where an example could be 
```python
payload = lambda x, y, z: x**2 + y**2 + z**2
```
applied across the `parameter` dimension.


The ``fluent`` module provides a ``from_source`` method for creating the initial node array by specifying an array of functions. For example to create a single node that opens a xarray dataset
```python
import xarray as xr 
import functools
from cascade.fluent import from_source

func = functools.partial(xr.open_dataset, "/path/to/dataset")
initial_action = from_source(func)
```
To create multiple nodes, supply an array of functions to be executed in each node, and optionally dims and coordinates for labelling the axis of the node array. For example, 
```python
import numpy as np 
import functools
from cascade.fluent import from_source

func = functools(np.random.rand, 2, 3)
initial_action = from_source(np.full((2, 2), func), dims=["x", "y"])
```
In this case, `initial_action` would contain a (2, 2) array of nodes with dimension "x" and "y", each containing (2, 3) random numpy array. 

One can then construct a graph from this point using the function cascading API:
```python
from cascade.fluent import from_source

graph = (
    from_source(np.full((2, 2), func), dims=["x", "y"])
    .mean("x")
    .min("y")
    .expand("z", internal_dim=1, dim_size=3, axis=0)
    .map([lambda x, a=a: x * a for a in range(1, 4)])
    .graph()
)
```

Resources 
---------

The graphs constructing using the Casacde API do not contain any annotations for resource usage of the tasks in each node. To manually attach resources to the tasks, we transform the graph into a ``TaskGraph`` object, providing a dictionary of `Resources`, containing CPU cost and memory, for each node name. 
```python
from cascade.transformers import to_taskgraph
from cascade.taskgraph import Resources

task_graph = to_taskgraph(graph, {
    x.name: Resources(100, 50) for x in graph.nodes()
})
```
Alternatively, the graph can be executed and profiled to determine the duration and memory requirements of each task using memray.
```python
from cascade.profiler import profile 
from cascade.executors.dask import LocalDaskExecutor

executor = LocalDaskExecutor()
results, annotated_graph = profile(graph, "/path/to/memfiles/dir/", executor)
```
The profiling returns the results of the graph execution as well as a `TaskGraph` containing the nodes of the original graph annotated with the duration and memory usage profiled.

Schedulers
----------

The schedulers assign the tasks in the graph to workers according to a ``ContextGraph`` and the resource requirements of the tasks in the ``TaskGraph``. Currently, two schedulers are available:
- ``DepthFirstScheduler``: assigns tasks according using a similated run of execution proceeding via a depth
first traversal of the graph
- ``AnnealingScheduler``: schedules the graph using a simulated annealing algorithm, with the initial 
schedule determined by the ``DepthFirstScheduler``


Executors
---------

The available executors are based on the Dask distributed local executor and Kubernetes executor. Both have been adapted to either use the dynamic scheduler provided by Dask or execute according to a provided schedule. 

**Does not currently create workers according to ``ContextGraph``** 

To execute a graph using the ``DaskLocalExecutor``, do the following:
```python
from cascade.executors.dask import DaskLocalExecutor

results = DaskLocalExecutor(n_workers=2, threads_per_worker=1, processes=True,
    memory_limit="10G").execute(graph)
```
The ``graph`` variable can be ``Graph`` or a ``Schedule``. If a ``Schedule`` is provided, then the 
executor will modify the task graph in order for Dask to execute it according to the predefined 
schedule. The method returns a dictionary of node names and associated contents of the sinks in the 
graph.

The execution also produces a Dask performance report, which is a html file containing information about the resources used during the run and a task stream for each worker. 

