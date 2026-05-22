# Selected Features in Cascade

This document explains selected features in Cascade that distinguish it from other workload executors, could be considered innovative, or exploit some specific of the domain it is intended for.
In other words, this could be read as "why didn't you guys use Dask?".

## Heterogeneous environment
When we are running a DAG, individual nodes are executed in different processes, possibly on different hosts.
As those serialize and deserialize objects to facilitate data flow from sources to sink, it is imperative that the environments (python venvs) are compatible.
Other solutions reach this by enforcing the same docker image, or the same virtual env, being used for a particular DAG or even for the lifetime of a cluster.

This, however, poses a barrier -- for example, if we want to run a DAG with multiple ML models (whether ensemble or coupled), each trained with a different version of libraries.
Or if we want to minimize the venv/image size, and download some bulkier libraries (like torch) only when needed.

We chose a solution of allowing any pip install per *DAG node*, thus giving maximum flexibility.
This is technically done by all processes on a given host being launched with one (parent) venv, but able to run `uv pip install --prefix {temporary_directory_per_node}`, and having that temporary directory added to `sys.path`.
Given `uv`'s ability to create `venv`s at fast pace as well as hardlinking cached dependencies, this gives a very negligible overhead.
See `cascade.runner.package` for technical details.

This of course gives the users the option to make a broken DAG, such as declaring a node with `numpy<2` and a node with `numpy>=2`, and expecting the serde to work correctly.
However, we tend to use stable data contracts and formats, so don't expect this to be a major problem.
Additionally, in many cases such as numpy or arrow we don't rely on pickling/unpickling, but instead pass the underlying buffers -- this gives us additional performance (in combination with the following section on Shared Memory).

## Shared Memory
An ideally paralellizable job has the data separable into disjoint partitions such that each compute requires exactly one partition.
Other solutions perform best when the DAGs in question are consisting primarily of bunch of ideally paralellizable jobs, interleaved with some shuffles or partition coalescing.

However, we find that our tasks often don't have this property.
To put very crudely, we can't separate the globe into e.g. continents and forecast weather per continent -- everything relates to everything.
Thus, very often we have DAG nodes that are computationally independent but that require the exact same data.

Due to the (current) nature of python, most solutions are paralellizing by spawning a number of processes (in the order of CPU cores) and then distributing data among them.
If we were to do that, we risk having a dataset in RAM as many times as there are CPU cores, asking for an Out-of-Memory ticket to coredump.
Instead, we utilize `multiprocessing.shared_memory` from python standard library (dubbed "POSIX style").
This allows us to have a single copy of a dataset in the RAM of the host, accessible by all processes, without any locking (as we enforce read-only).

This has other advantages.
Firstly, it allows us to pass data from one DAG node to another (whether on the same host or a different one) without bothering the producer DAG node for long.
It simply writes to shared memory and continues with other computations, while cascade agent handles the data transfer to a different host.
The DAG node thus does need to live longer than needed to just e.g. retry the data sending, waiting for the co-located process to start and ack recepetion, etc.
Secondly, it gives us additional robustness -- processes calculating DAG nodes can crash for e.g. OOM reasons.
But no important data is lost with any process crash, because the data remain persisted in the shared memory.

We additionally manage the shared memory occupation, moving to disk if under pressure (a smarter solution would perhaps utilize kernel swapping implicitly).
But that's just to prevent future OOMs, not to facilitate checkpointing.

Note that this is not intended for dynamic message exchange between two coupled tasks in the MPI style -- for that it would not perform.
Our workflows handle this using eg `nccl`, `gloo` or `mpi` -- but those are not really handled or enforced by Cascade, except for providing the group's addresses to each member, to facilitate the communication bootstrapping.

For technical details, see `cascade.shm`.

## Generator Tasks
Other solutions are based on the premise of "at the end of each DAG node, its output is collected and provided to its DAG children".
However, we often have ML models which generate a number of outputs (such as forecasts for 1, 2, 3, ... days), and each of their outputs can be consumed by a single DAG node as an independent data partition.
We cannot de-serialize the ML model (as it is auto-regressive in essence), but we would like to start the downstream tasks as soon as possible.

We thus allow our DAG nodes to publish outputs while they still run, and have scheduler react not to DAG node completion, but to every single dataset publication.

Some of the other solutions allow implementation of this using agent concepts -- but we find agents an impractical abstractions.
They fit well a role of a long-lived entity with arbitrary number of message-like inputs and outputs, whereas we deal with a regular DAG node -- it just has more outputs, and with eager publishing.

## Sophisticated Scheduling
All technical details for the following sections can be found in `cascade.scheduler` module.

### Lookahead
Cascade is intended for non-interactive DAG execution, where the whole DAG is known in advance, including information such as whether a particular DAG node needs a GPU or not.
We utilize the information in the scheduler, trying for example to assign parents of a GPU-requiring DAG node to a GPU-powered host even if those parents don't need a GPU themselves, to reduce the amount of cross-host transfers.
We also don't need to broadcast datasets -- we only transfer dataset to where it is needed, and in time prior to actual computation starting.

Other solutions provide simultaneously a good interactive experience as well as dynamic graph definition, which is a noble and beneficial goal, but complicates life tremendously.

### Profile-Awareness
In addition to the DAG being known in advance, we also know that the resource requirements (memory, cpu time) will remain more or less stable.
That is again given to us by the domain -- forecasts today operate on the same volume of data and with the same computation as they will do tomorrow.
The data themselves obviously change, and model weights may potentially too, but the shape is constant, at least between major version release cycles.

The scheduler can utilize this in multiple ways.
Firstly, we can make a guess when a particular computation on a host will finish, decide what the next task will be, and start sending datasets (if they are on a different hosts) in advance -- this was already mentioned in the previous section.
Secondly, we can tell whether a particular host would handle a given computation in terms of RAM capacity, and thus can prevent an OOM.

Our scheduler is actually a combination of a static one (we analyze the DAG structure and profile information once, before any execution starts) and a dynamic one (to account for variance in network traffic speed, noisy neighbour effects, et cetera).
What can get precomputed will be, but we know well that no profile can be fully trusted.

Other solutions would not benefit from such feature, because they are often used for DAGs with varying data amounts -- a job that counts webpage visits has a very different profile over time.

### Task Fusing
Let's imagine a simple DAG which is just a path.
In this case, there is no better workload executor than `for` cycle.
We aim to get closer to this ideal state by fusing tasks, that is, bundle multiple DAG nodes into a single `for`-cycle executable package.
This minimizes the amount of scheduler-executor messages as well as reduces pressure on the shared memory system.

We don't however apply only simple edge contractions.
Instead, we seach for heaviest (in terms of data volume) source-sink paths in the DAG, and mark those as fuseable entities for the scheduler to pick.
During execution, the scheduler *may* decide to fuse those, depending on other input availability and output requirements.
We utilize the publish-multiple-outputs feature we introduced in the "Generators" section to facilitate fusing even when an output of some DAG node in the middle of the fused path is required by other DAG nodes.

This leads to healthy scheduling patterns where we observe long sequential data-heavy paths executed by individual hosts, with occassional interchange or synchronization happening if the DAG is particularly interconnected or branching.

We are not aware of other solutions implementing this optimization -- but it is rather important for us.
Note lastly that this allows us to keep data in GPU across DAG nodes.
As our DAG nodes are rather fine-grained (think thousands of nodes), without this optimization the majority of time would be spend on the GPU back-and-forth.

## ZMQ
We chose ZMQ as our communication layer of choice, giving us better performance than for example http would.
However, we perceive the biggest advantage in the design it enables: there is no blocking communication whatsoever, only sending messages and commands, and threads polling multiple sockets.
And that comes without any requirement like distinguish whether a function is `async` or not.
Refactoring the communication topology or patterns is relatively easy, as any coupling is very low -- we have done so multiple times already.
And both scheduler and executors allow naturally for a reactive design.
There is no distinction between how the scheduler, the executor agent (one per host that manages the shared memory data and distribution), and the individual executing processes (many per each host) communicate -- all of them message for _all_ mutual communication, and the implementation of ZMQ handles which of the tcp/ipc/inproc routes is chosen.
Lastly, ZMQ frames allow us for low-memory-overhead transfer of large datasets -- if done naively, a dataset may live in memory up to three times when being transfered (once in executor, once in shared memory, once in ZMQ), whereas we manage one copy plus a single-frame overhead.

Other solutions use for example Tornado, but we found ZMQ more performant and stable.

For technical details, see `cascade.executor.comms`.
