# Cascade High Level Interface

## Intro
Let's first make sure we get our data representation right -- designing API on top of that is usually easier.

We have already clarified a low level cascade representation:
 - task instance / node object essentially wraps a python callable -- can run in a single process, fits in a single machine, completely executable
 - job instance / graph object ties task instances together in a way that a scheduler can schedule and executor can execute

This representation is optimised for executor (all information present) and for scheduler (broken down into atomic pieces), but neither for human reading nor human writing.
We thus need a higher level object hierarchy, which we could lower into this.

### Criteria:
 - _reliability_ -- no class cast exception in the middle of graph execution, no dependency version mismatches, no venv mixups, portability.
 - _extensibility_ -- we don't want to be tied to any particular data format, but should instead support in extensible fashion: numpy, earthkit, xarray, grib, ... A cascade graph must be able to combine _distinct_ data sources, possibly with explicit conversion, e.g. `xarray_action = cascade.from_source(load_xarray).f1(); earthkit_action = cascade.from_source(load_earthkit).f2(); result = some_join(xarray_action.to_earthkit(), earthkit_action)`
 - _developer convenience_ -- we assume a few developers writing a large number of jobs relying on wide range of repos, this should be effortless and boilerplate-free.
 - _familiarity_ -- scientists and other non-cascade developers should not face a steep learning curve when authoring _basic_ cascade jobs.

### Current State
 - We do not have means of plugging in metadata needed for reliable execution (env versions, types).
 - Extensibility presumably clashes with familiarity/convenience, because we define user functions on children of Action class, which causes a need for the `.switch` action that does in no way modify the graph itself

### Competition Analysis:
There are no native ways of
1. extending Spark/Dask/Xarray data objects with functions: you can `def my_f(df)`, but you can't `df.my_f`.
2. combining frameworks -- you can't have a distributed job which mixes Dask dataframe and Spark dataframe. You have to decide on your data represantion at the beginning.
3. analyse graph correctness during building for type safety.
4. override environment for reliable execution on single task level.

## High Level Representation
We currently have an "Action" object, but it's semantics are a bit fuzzy / open.
One option is to rename it "CascadeDataset[T]", and have it consist of:
 - T: being of numpy, xarray, earthkit, ... T is fixed for a single CascadeDataset, but the whole graph can contain multiple Ts
 - metadata of the dataset: in particular, dimensions/partitions, and an enum representing T (for access during graph building)
 - action: describing _how_ this dataset is computed, eg. by `map(parent, myF, dim)`, or `concat(parent1, parent2, dim)`

This yields multiple automated derivations:
 - CascadeDataset expands into TaskInstances as follows:
   - based on the dimension of the action and T, we generate TaskInstances with callable being `select` or `stack` etc that prepare the ground / clean up
   - based on the param of the action and T, we generate TaskInstances doing the logic, e.g. callable being `ds.map(myF)`
 - metadata consistency can be checked -- the metadata of parent, when transformed with action, yield this
 - environments of TaskInstances can be generated -- T gives a core dependency, and the map's payload optionally brings its own one

This is still not conveniently writeable, but at least matches 1-1 to what user would be writing -- we'd love to see:
```
get_data(how)
    .operation1(fewParams)
    .operation2(noParams)
    .operation3(fewParams)
    .operation4(output)
```
that is, every _line_ in the above corresponds to a single `CascadeDataset[T]`, but in a way that all the T and metadata and action are derived from what the user puts in.

### Extensibility and Repo organisation

#### A new T
We believe that the cardinality of T would be reasonable -- so adding a new T needs not to be a 5-minute chore.
A day or two instead is legit.
Also, there is a set of operations that would be common for all T'-- map, reduce, project (select), stack (concatenate), convert as primitives, and e.g. mean or threshold or quantile being higher level.
We would define those operations on the base class CascadeDataset, and during lowering into TaskInstances the right callables/entrypoints would be looked up from the T (not annotation, but field!Remember we keep it).
To implement a new T would then amount to provide a "dictionary" of for "CascadeGraph.sel(dim) on this means TaskInstance(entrypoint=xarray.Dataset.sel)", etc.

In principle, all these could live in core cascade, because the dependencies themselves (xarray, earthkit) are not needed during the graph building in this fashion.
Ideally, we'd have all TaskInstances defined via entrypoint (string), not via callable (cloudpickled bytes).
It may happen that not all action we'd like to express would have a viable entrypoint in eg xarray -- then, we'd have to write xarray-cascade-ext wheel, and use _that_, instead of xarray, as a core environment for xarray-based TaskInstances.
But this dependency would not needed to be known to the authors of (basic) cascade graphs.

#### A New Operation
Say the user is happy with earthkit, but want to define their own high level operation.
The following are conceivable:
1. A composition of existing actions, `my_composite = lambda ds: ds.sel(param=2t).threshold(.8).mean(over=member)` -- the user just wants reusable/readable code.
2. An UDF passed to an existing action -- `my_cascade_dataset.map(myCustomXarrayFunction)` -- the user has some domain-specific business logic.
3. A wholly new _kind_ of action, something not expressible via map/reduce/project/stack. For example, `my_cascade_dataset.partitions_as_iterator()`.

Number 1 has no change on representation -- is just a matter of expressing compactly.
Number 2 _only_ needs the user to either supply a Callable itself (presumably in interactive mode), or `[pipPackage]module.entrypoint` string (presumably in release mode).
Neither would allow us to do a good typechecking -- we can't inspect signatures at the dimension transformation level.
For interactive mode, this is acceptable (to a regular Python developer at least) -- for a release mode either we call it Beta, or we manually supply the information if that is an important product.
Number 3 I'm going to assume it's rare as well as non-trivial/non-generalizable, so fundamental work by a cascade developer is justified.

It is assumed that 1 & 2 are actually very frequent -- all current the cascade-pproc and cascade-anemoi etc contain or will contain loads of business logic and convenient aliases that need to be readily accessible.

Speaking of this example for Number 3 -- we could actually support Iterators neatly here by extending the dimension object: instead of `{step: [1, 2, 3], param: [2t], ...}` for all-at-once prediction (current state) you'd have `{step: IteratorDimension([1, 2, 3], ...`, and the `project/select` implementation would be able to handle this.

### User-facing API
Let's ideate how would user create this high level representation conveniently.

#### Non-fluent API
```
from cascade.backends import earthkit
from cascade.pproc import temperature_mean_threshold
from cascade.anemoi import invokeModel
from cascade import cascade

raw_mars = cascade.from_source("mars", earthkit, mars_params)
# That translates to CascadeDataset(T=Earthkit, dimension=from(mars_params), action=(mars.query, mars_params)

model_outputs = invokeModel(raw_mars, model_checkpoint, steps)
# That translates to CascadeDataset(T=Earthkit, dimension=parent.dimension + steps, action=([aifs-inference]anemoi.run, args=(model_checkpoint, steps), parent=parent))

postproc = temperature_mean_threshold(model_outputs, pproc_params)
# CascadeDataset(T=Earthkit, dimension=[2t], action=([cascade-pproc-impl]temperature_mean_threshold, args=(...)

display(postproc)
# CascadeDataset(T=png, action=[earthkit.plots]display_map, ...)
# we may want to distinguish CascadeDatasetRich (for T=xarary, earthkit) and CascadeDatasetPrimitive (for T=png, str), where the latter would support only small subset of ops/children
```

This looks reasonably neat for the user -- you don't need to provide anything beyond what you would expect.
There is some overhead for the developer of the functions like `invokeModel`, `temperature_mean_threshold` -- you need not just implement the function itself, but also the "cascade interface" of it -- the collection of `[package]module.entrypoint` string and (optionally!) information for deriving the dimensions.
Those cascade interfaces could presumably also live in the `cascade` core itself -- they won't runtime depend on anything.
If we'd worry about repo polution / business logic overload, we can totally have `cascade.pproc` etc be their own repos.

#### Fluent API
The primary problem is how to get all the `cascade.pproc`, `cascade.anemoi` functions to _register_ into the fluent builder object.
Once that happens, the above just translates into:
```
cascade.from_source(...)
    .invokeModel(...)
    .temperature_mean_threshold(...)
    .display(...)
```
ie, just drop intermediate vars and `s/=/./`.

I can ideate these _dynamic_ implementation approaches:
 - override getattr to dynamically look up module and inspect it: then `some_cascade_action.pproc.temperature_mean_threshold` would first notice there is no `pproc` member of the base class, so it would try to import some module like `cascade.pproc` and return a class with _also_ overridden getattr to resolve the function itself (in the right wrapper)
 - have the modules like `cascade.pproc` upon import register into some singleton object which the base fluent builder would look into.

Both work, both have upsides and downsides:
 - typing `.pproc` or `.anemoi` etc is overhead, but perhaps adds clarity and resolves conflicts,
 - the register-upon-import _needs_ the import, otherwise it fails during graph building,
 - the lookup-based-on-prefix _needs_ a reliable module naming convention,
 - neither effectively supports linting, hinting, mypy (this one I believe I _could_ make work, but I don't think its worth the effort).

And one _static_ implementation approach -- we generate a class containing all the functions. This is essentially doable only in one-repository scenario -- with multiple repositories, it would perhaps need to be done in some post-install hook and that has fragile vibes.

#### Other noteworthy cases
Harrison's example:
```
Take a data source from anemoi, split across param and time
Cache some of the params
With all of the data:
    Switch to a pproc action
    Run a pproc function
Change environments to earthkit
Join the climatology
Run a base cascade mean
Map a custom function across the time dim
```
... and I think the above supports all of it :), perhaps like
```
raw = cascade.from_source(anemoi_input_param_spec)
anemoi = invokeModel(raw, model_params).select({param: [2t, q, u, v]})
interestingParamsCache = anemoi.select({param: 2t, step: [2, 4})
postproc = pprocPipeline(anemoi, pproc_params).convert(cascade.earthkit)
climatology = cascade.from_source(climatology_input_param_spec)
combined = cascade.stack([postproc, climatology], dim={"source": ["model", "climatology"]})
result = combined.mean(dim=["source", "time"]).map(myCallable, dim=["time"])
```
Which is a hybrid between non-fluent and fluent -- in part justifiable, because we have two sources and two sinks!
I wrote it as I write my pandas etc scripts -- they are typically also a mixture.
