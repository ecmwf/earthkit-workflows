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