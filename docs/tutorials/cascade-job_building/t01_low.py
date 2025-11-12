"""Build a job using the most low level interface possible. This will be verbose and unwieldy,
but exposes all the available toggles

For start, we want to generate data, then calculate mean of a single variable over time
these two conceptual tasks translate into three computational steps:
1. generate data
2. project into the variable in question
3. calculate the mean
We will thus build a graph consisting of these three tasks

Each task requires primarily the callable with the actual python code,
and then some metadata about how to glue things together
"""

import numpy as np
import xarray as xr
from t00_execute import run_job
from t00_generate import generate_data

from cascade.low.core import (
    DatasetId,
    JobInstance,
    Task2TaskEdge,
    TaskDefinition,
    TaskInstance,
)

task1_callable = generate_data


def task2_callable(d: xr.Dataset) -> np.ndarray:
    return d["precip"].to_numpy()


def task3_callable(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=(0, 1))


task1_definition = TaskDefinition(
    func=TaskDefinition.func_enc(task1_callable),
    environment=["xarray"],  # any pip installable thing
    input_schema={},
    output_schema=[("0", "xarray.Dataset")],
    needs_gpu=False,
)
task2_definition = TaskDefinition(
    func=TaskDefinition.func_enc(task2_callable),
    environment=["xarray"],
    input_schema={"d": "xarray.Dataset"},
    output_schema=[("0", "numpy.ndarray")],
    needs_gpu=False,
)
task3_definition = TaskDefinition(
    func=TaskDefinition.func_enc(task3_callable),
    environment=["numpy"],  # we dont need xarray anymore
    input_schema={"a": "numpy.ndarray"},
    output_schema=[("0", "numpy.ndarray")],
    needs_gpu=False,
)

# one task definition can be used multiple times with different static parameters,
# so theres one extra wrapping
task1_instance = TaskInstance(
    definition=task1_definition, static_input_kw={}, static_input_ps={}
)
task2_instance = TaskInstance(
    definition=task2_definition, static_input_kw={}, static_input_ps={}
)
task3_instance = TaskInstance(
    definition=task3_definition, static_input_kw={}, static_input_ps={}
)

# now we glue tasks together
edge1to2 = Task2TaskEdge(
    source=DatasetId("task1", "0"),
    sink_task="task2",
    sink_input_kw="d",
    sink_input_ps=None,
)
edge2to3 = Task2TaskEdge(
    source=DatasetId("task2", "0"),
    sink_task="task3",
    sink_input_kw="a",
    sink_input_ps=None,
)

# and put everything together
job = JobInstance(
    tasks={"task1": task1_instance, "task2": task2_instance, "task3": task3_instance},
    edges=[edge1to2, edge2to3],
    ext_outputs=[DatasetId("task3", "0")],
)

if __name__ == "__main__":
    print(run_job(job))

# output will look something like this
# compute took 1.505s, including startup 1.559s
# {task3.0: array([0.46984753, 0.486206  , 0.46872369, 0.54454383, 0.4908007 ,
#        0.48240363, 0.51079899, 0.480964  , 0.51560131, 0.48102477,
#        0.49124097, 0.48426894, 0.48035238, 0.48734304, 0.48078968,
#        0.49163688, 0.45139265, 0.47323565, 0.51291645, 0.52455964,
#        0.49186845, 0.44741059, 0.5159331 , 0.47815734])}
