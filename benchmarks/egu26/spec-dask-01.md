I want you to implement two python scripts, each showcasing a particular computation in the Dask framework in python.
The role of the script is to serve as a benchmark baseline.
The computations inside the Dask won't have any particular value, they are just to manifest some activity.
Thus, do not attempt to optimize them away.
Write it as intermediate Dask user -- it is not to exploit or utilize hidden or advanced Dask properties, but also don't be purposefuly suboptimal.
Each benchmark should be a single python file, which takes care of spinning up a local dask cluster, submitting the Dask task to it, awaiting its completion, and reporting success or failure.

The definition of benchmarks:
First benchmark showcases "Single Data, Multiple Instructions". The dask graph should consist of one source node that outputs a numpy matrix (uniform random [0,1]), and then a number of child nodes, each of which consumes the same matrix and computes some operation on it: use some standard functions, like power of the matrix, eigenvalues, trig functions, ... 
The benchmark must be parametrizable by the following:
- size of the matrix, `N*N`: the dimension N comes from `BENCHMARK_N` envvar,
- number of children tasks, coming from `BENCHMARK_M` envvar. We want this to be deterministic, so fix an ordering of the child functions, and each child has an index I and computes the function I % `number_of_functions`

The second benchmark showcases "Batch Data Generation". The dask graph should consist of one source node that `yield`s numpy matrices (uniform random) and sleeps for T seconds, a second node that consumes a list of matrices, and for each computes the SVD and sums the resulting singular values (nuclear norm), and a third node which consumes a list of those floats and outputs their sum.
This source should be inherently sequential -- so lets say it keep the last yielded matrix, and each time it generates a new one, it actually yields their product.
There are parameters `BENCHMARK_N` for matrix size and `BENCHMARK_M` for the number of matrices to be yielded, and `T` for the sleeping.
The baseline implementation is "wasteful" in that it allows no eager concurrency.
You can provide a second implementation, that utilizes Dask actors for this -- however, this must still preserve the sequentiality of the generator. Two non-obvious pitfalls: (1) the SVD must be computed outside the actor (in a regular submitted task), otherwise the actor worker is busy with SVD and no pipeline overlap happens; (2) `client.submit` deduplicates by default when function and arguments are identical, so submitting the same actor-calling function M times collapses to one task -- pass `pure=False` to prevent this.

After you implement the benchmarks, make sure you can run them with some basic values of N and M, like 10 and 10.
