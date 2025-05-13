# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A completely artificial job to showcase a use of Generator in the output of a task

There is one source which belches out N matrices of size 2**K, and the consumer tasks
just compute their Lth power
"""

import os
from typing import Iterator

import numpy as np

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance, TaskDefinition, TaskInstance, type_enc


def ser_numpy(a: np.ndarray) -> memoryview:  # bytes:
    """Exists just because numpy.ndarray cant be imported"""
    # return a.tobytes() # beware, this includes a big copy
    return a.data.cast("B")


def get_job() -> JobInstance:
    N = int(os.environ["GENERATORS_N"])
    K = int(os.environ["GENERATORS_K"])
    size = (2**K, 2**K)
    L = int(os.environ["GENERATORS_L"])

    def generator() -> Iterator[np.ndarray]:
        for i in range(N):
            yield np.random.uniform(size=size)

    def consumer(i: np.ndarray) -> np.ndarray:
        return np.reshape(i, size) ** L

    generator_d = TaskDefinition(
        func=TaskDefinition.func_enc(generator),
        environment=[],
        input_schema={},
        output_schema=[(f"{i}", "ndarray") for i in range(N)],
    )
    generator_i = TaskInstance(
        definition=generator_d, static_input_kw={}, static_input_ps={}
    )

    builder = JobBuilder()
    builder = builder.with_node("generator", generator_i)
    for i in range(N):
        builder = builder.with_node(f"consumer{i}", TaskBuilder.from_callable(consumer))
        builder = builder.with_edge("generator", f"consumer{i}", "i", f"{i}")
    job = builder.build().get_or_raise()
    job.serdes = {
        type_enc(np.ndarray): (
            "cascade.benchmarks.generators.ser_numpy",
            "numpy.frombuffer",
        )
    }
    return job
