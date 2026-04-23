Now we will recreate the benchmarks using cascade, the project of this repository.
Inspect the integration_tests at the root of this repository: the `job_runnerRestarts.py` should show you how to define a basic cascade job, and the `harness.py` how to execute them.
We haven't had any settings for the cluster size for dask yet, so we will need to fix that -- lets say `BENCHMARK_W` will define the number of dask processes, and then for cascade the number of workers (there will always be one host).

You will need to split the files a bit -- lets say we will have `sdmi_{runtime, dask, cascade}.py` files, where the dask/cascade will be the entrypoint that each imports from the respective runtime; the same for `bdg` (except that it will have `dask_baseline` and `dask_actors` entrypoints).

The `sdmi` should translate directly, dask tasks being cascade tasks.
The `bdg` should contain only one implementation for cascade -- to learn how to utizile generators, inspect `integration_tests/benchmarks_old/generators.py` which implements already a very similar job.
