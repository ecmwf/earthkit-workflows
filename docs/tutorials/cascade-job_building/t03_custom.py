"""So builders are neat but with task per dimension x chunk x computation, thats still a lot to type

And if you want to "just apply mean", you need to handle all the edgy business of what glues to what,
the projections, the concatenates, ...

We thus need something that ideally both captures some business specific logic *and* operates on subgraphs,
rather than on individual tasks -- like "take whatever xarray.Dataset comes your way and calculate mean
over this coordinate, regardless of other variables, partitions, chunkings, ..."

Let us try to build a very simple framework. We will have classes corresponding to subgraphs: class for
reading data, class for applying a numpy function, ... Each such class could be glued to another class
in a more convenient fashion than we had before, and each such class would expand into multiple cascade
tasks, ie, a subgraph. We will identify each subgraph with the xarray.Dataset it exposes for further
computation, but we will hide the details of how many partitions (tasks) it consists of.

This implementation is a toy only! It shows a bare minimum how to implement a framework like earthkit.workflows.fluent,
which in itself is more rich and convenient. But ultimately everything ends up producing cascade.low.core.JobInstance
"""

import uuid
from typing import Callable, Iterator, Self

import numpy as np
import xarray as xr
from t00_execute import run_job
from t00_generate import generate_data

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance

"""
This is the base class for us -- we hold the `implementation` which is the JobBuilder so far,
ie, the graph leading up to and including this subgraph, and the "xarray.Dataset" which is offered

There are no methods to be overridden at the children, everything is supposed to happen during init
"""


class DatasetSubgraph:
    # graph
    implementation: JobBuilder
    # xarray.Dataset -- we need just the variable/dims metadata, we dont want to build the actual instance
    dims: list[str]
    variables: list[str]
    projection: dict[str, set[str]]  # which graph node holds which variables

    def as_instance(self) -> JobInstance:
        return self.implementation.build().get_or_raise()


"""
Now we will implement two children of the base class, each representing a different kind of computation:
* Source -- produce a DatasetSubgraph from nothing
* Apply -- produce a DatasetSubgraph from another DatasetSubgraph, given an f and params to apply
"""


class Source(DatasetSubgraph):
    """Creates a no-inputs DatasetSubgraph using the provided `kallable`"""

    @staticmethod
    def _validate(i: xr.Dataset, dims: list[str], variables: list[str]) -> xr.Dataset:
        # this is just a method that runtime-validates metadata consistency with actual data
        assert set(dims) == set(i.dims.keys())
        assert set(variables) == set(i.keys())
        return i

    def __init__(
        self, kallable: Callable, dims: list[str], variables: list[str]
    ) -> Self:
        self.dims = dims
        self.variables = variables
        # note that now we have two tasks here! One that reads the data, another that validates
        self.implementation = (
            JobBuilder()
            .with_node(
                "read", TaskBuilder.from_callable(kallable, environment=["xarray"])
            )
            .with_node(
                "validate",
                TaskBuilder.from_callable(Source._validate).with_values(
                    dims=dims, variables=variables
                ),
            )
            .with_edge("read", "validate", "i")
        )
        self.projection = {"validate": set(variables)}


"""
The Apply class is a bit more complicated, because we can't assume much about our input.
We know it is a DatasetSubgraph class, so xr.Dataset spread across multiple cascade tasks.
We thus need to inspect the input (at graph building time!), determine which variable to get
from where, and build all the tasks for individual `apply`.
We also need to set the resulting dims/variables correctly, ie, read what we get at the input,
ideate (at graph build time!) what will be the result of our numpy apply later on during
execution, and set it -- so that a potential follow-up Apply has correct metadata to work with.
"""


class ApplyNumpyReduce(DatasetSubgraph):
    """Reduces `over_dims` for each `over_variables` individually via `f`"""

    @staticmethod
    def _apply(f: Callable, i: (str, xr.DataArray), over_dims: list[str]) -> xr.Dataset:
        # the core apply method -- we already assume the input is a single variable
        varName, array = i
        axis = tuple(idx for idx, e in enumerate(array.dims) if e in over_dims)
        raw = f(array.to_numpy(), axis=axis)
        remainingDims = [e for e in array.dims if e not in over_dims]
        result = xr.DataArray(
            raw, dims=remainingDims, coords={d: array.coords[d] for d in remainingDims}
        )
        # we need to wrap so that possible downstream actions can parse
        return xr.Dataset({varName: result})

    # there are two project methods, required because of how cascade/python act on generator functions. I need to fix this so that only the generator is required
    @staticmethod
    def _project_one(i: xr.Dataset, variables: list[str]) -> tuple[str, xr.DataArray]:
        return variables[0], i[variables[0]]

    @staticmethod
    def _project_gen(
        i: xr.Dataset, variables: list[str]
    ) -> Iterator[tuple[str, xr.DataArray]]:
        for v in variables:
            print(f"yielding for {v}")
            yield v, i[v]

    def __init__(
        self,
        over_variables: list[str],
        over_dims: list[str],
        f: Callable,
        frum: DatasetSubgraph,
    ) -> Self:
        # this is where the input-parsing and output metadata derivation happens

        assert set(over_dims) <= set(frum.dims)
        assert set(over_variables) <= set(frum.variables)

        self.variables = frum.variables
        self.dims = [e for e in frum.dims if e not in over_dims]
        self.implementation = frum.implementation
        self.projection = {}

        for source, variables in frum.projection.items():
            if target := [v for v in variables if v in over_variables]:
                kallable = (
                    ApplyNumpyReduce._project_one
                    if len(target) == 1
                    else ApplyNumpyReduce._project_gen
                )
                projectT = TaskBuilder.from_callable(kallable).with_values(
                    variables=target
                )
                projectT.definition.output_schema = [
                    (f"{i}", "tuple[str, xr.DataArray]") for i in range(len(target))
                ]
                projectN = f"project_{uuid.uuid4()}"
                self.implementation = self.implementation.with_node(
                    projectN, projectT
                ).with_edge(source, projectN, "i")
                for i, v in enumerate(target):
                    applyT = TaskBuilder.from_callable(
                        ApplyNumpyReduce._apply
                    ).with_values(f=f, over_dims=over_dims)
                    applyN = f"apply_{uuid.uuid4()}"
                    self.implementation = self.implementation.with_node(
                        applyN, applyT
                    ).with_edge(projectN, applyN, "i", f"{i}")
                    self.projection[applyN] = {v}


"""
And that concludes the "framework" implementation, now on to writing some jobs with it!
"""

# this is basically the job we have implemented in the previous tutorial steps
source = Source(generate_data, ["x", "y", "t"], ["precip", "temper"])
precipMeanOverTime = ApplyNumpyReduce(["precip"], ["x", "y"], np.mean, source)

# but we can now parametrize easily
temperMaxAtPlace = ApplyNumpyReduce(["temper"], ["t"], np.max, source)

# or even chain operations
temperMeanOfMaxes = ApplyNumpyReduce(["temper"], ["x", "y"], np.mean, temperMaxAtPlace)

# or process two variables within a single job
meanForEachVariableOverTime = ApplyNumpyReduce(
    ["precip", "temper"], ["x", "y"], np.mean, source
)

if __name__ == "__main__":
    # print(run_job(temperMeanOfMaxes.as_instance()))
    print(run_job(meanForEachVariableOverTime.as_instance()))
    # NOTE you can observe the execution graphs in the logs -- in the temperMeanOfMaxes,
    # everything gets fused into a single computation on a single worker, because
    # the tasks are a linear sequence. But in the meanForEachVariableOverTime, we'll get
    # two workers utilized -- one handles a (fused) package of reading data and calculating
    # one variable, but the other kicks in once the data is read and computes the other
    # variable.
    # How can this be read from the logs? Look for message like
    # `sending TaskSequence(worker=h0.w1, tasks=['read', 'validate', ...`

"""
(thought) exercises for the reader
1. ApplyNumpyReduce assumes reduction in coordinates -- what would we need to do to allow for an apply that preserves dims?
2. We dont set ext_outputs anywhere, meaning nothing is returned when we run the job! How would we best go about it?
3. Imagine we want a job that outputs temperMeanOfMaxes *and* precipMeanOverTime at once, ie, the Source is shared -- what would we need to change?
   note: try to come up with two distinct solutions here, hint: "merge" vs "singleton"
4. Implement a NumpyCombine, ie, a class that takes a callable and *two* DatasetSubgraphs, and the `f` takes two xr.DatasetArrays. For example, `sum`.
   note: each of the two distinct solutions from previous exercise can be used as a stepping stone here
5. Implement each of the precipMeanOverTime, temperMeanOfMaxes, etc, using the ekw.fluent module, ie, our mature solution of the framework problem
"""
