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



# Simulated Annealing Optimization

* Create a partition using a time sweep and depth-first search similar to Dask
* Run a simulated annealing optimization to improve the partitioning and ordering
    * Generate a neighbour, using temperature as a guide for the number/size of changes
        * Consider movement of subgraph to another node
        * Consider reordering of process subgraphs
        * Consider swapping of processing contexts for entire graphs
        * Consider duplicate computation of subgraphs
        * Reduce subgraph size as temperature decreases
        * Can be guided - i.e. move subgraph to a context with less work
    * Evaluate the neighbour, calculate the probably of accepting the neighbour
        * If the new partition is better, keep it
        * If the new partition is worse, keep it with probability e^(-delta/T)
    * Repeat until T is small
