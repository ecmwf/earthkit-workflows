# Key Technologies to Prototype

* cuda-python
    * aims to be close to the C API
    * requires describing kernels natively

* cu-py
    * easy to keep track of things on the GPU(s)
    * numpy operations automatically supported
    * custom kernels easy to define
    * zero-copy compatible with numba, pytorch
    * can export gpu memory locations to other libraries
    * "CuPy uses on-the-fly kernel synthesis. When a kernel call is required, it compiles a kernel code optimized for the dimensions and dtypes of the given arguments, sends them to the GPU device, and executes the kernel."
    * "It may take several seconds when calling a CuPy function for the first time in a process. This is because the CUDA driver creates a CUDA context during the first CUDA API call in CUDA applications."

* numba

* UCX
* UCX-py
    * supports CUDA through cu-py and numba


# Basic Design

* Worker processes execute on CPU -- one worker per e.g. GPU, CPU. May be multiple on each node.
    * Responsible for opening a read for each providing subgraph
    * Responsible for running the subgraph
    * Responsible for opening a send for each dependent subgraph
    * If direct send not possible, responsible for storage
    * Transport layer uses UCX-py
    * Workers could be long-lived of ephemeral

* Main manager responsible for spinning up workers and distributing subgraphs to them

* Each task defines a set of inputs, a function to execute, and a set of outputs

# Graphs

* Cascade should take, as input, a computational graph which describes tasks and data dependencies.
    * task, dependency on task_X, output_1
* It should split this graph into optimal subgraphs to be scheduled as single processes

# Graph Partionining for Scheduling

* Depth-first favours minimal memory consumption, outputs are consumed as a priority
* Breadth-first favours speed, critical path processes are started sooner
* We want to find a solution which is as breadth-first as possible without running out of memory

## Graph Partitioning by Merging

* Pick a processing context (fastest first), note its memory capacity M
* Compute the critical path (maximum time)
* Pick the node at the root of the critical path, N
* If not valid in the processing context, look for next critical path. Repeat as necessary.

* Scan the neighbours, searching for a node to add to subgraph
    * disregard nodes which cannot execute on the current processing context
    * disregard nodes which would exceed memory capacity of current processing context
    * weight based on inefficiency of processing if valid on multiple contexts
    * then weight based on communication cost -- how to judge cost? don't know remote exec context
    * and weight on contribution to critical path OR weight on processing cost (higher is better)
    * if equal, then by maximum memory usage
    * if equal, then by maximum processing cost
    * if equal, then by name
* Repeat until a maximum amount of memory is reached
* Store this as a subgraph, noting the expected runtime

* Choose the next processing context by looking for free processing contexts, or seeing which subgraph will complete first
    * store a 'total assigned time' of each processing context

* Choose the next node by finding the critical path of the remaining nodes, and pick the node at the root, repeat.

### Limitations

* Probably doesn't deal well with outputs becoming available in the middle of a sub-graph. No penalisation for outputs hanging around waiting for another context to receive them. Doesn't prioritise consumption of outputs (depth first).

## Graph Partitioning by Depth First then Expansion

* Find the output with the longest critical path
* Build backwards consuming as much of the critical path depth as possible
    * If processing context becomes full, from that edge
        * open a new context, continue building backwards
    * When an input node is reached

???

## Graph Thoughts

* The hardware can also be represented as a graph, but its probably not helpful
* Two graphs
    * a graph of tasks (with cost) with edges (with cost)
    * a graph of processing contexts (with capacity/speed) and edges (communication speeds)

## Graph Optimization

* To find an optimum solution probably requires representing the problem as some kind of optimization problem
    * Something to tweak
    * Something to measure

* Maybe we can optimise starting with a depth-first partition/ordering

* Optimise/brute force with work-stealing
    * Does it reduce TTS
        * by reducing waiting dependencies?
        * by increasing breadth?
    * Where there is a wait, try to steal from the other process?
    * May need to steal a whole subgraph to make it worth it
    * To evaluate the overall cost, need to consider the timeline of the whole execution

* Optimise/brute force with work duplication
    * Does it reduce TTS?

* Start from longest critical path output, work backwards minimizing memory cost

#### READ THIS

* Maybe use METIS/merging to coarsen graph into chunks
* Optimize by moving large chunks first, move to minimise wait/idle times
* Then smaller chunks

* Take inspiration from multi-grid solvers:
    * on very coarse level we could do a direct solve
    * move between contexts to optimise
