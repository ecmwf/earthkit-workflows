At the moment just an adhoc collection of scripts, utils and scenarios.

# Scenario: SHM Throughput
It seems that `AF_UNIX` + `SOCK_STREAM` and `AF_INET` + `SOCK_DGRAM` are, performance-wise, indistinguishable at the `shm` scale.
That leads us to prefer the former, because:
 - `SOCK_STREAM` is more reliable in general (though should not be noticeable in the localhost case),
 - it is easier to pick a random file name than a random *non-occupied* port number, and it's easier to free it after usage.

# Case Study: Scheduling with Sat Experiments
Using `ortools` -- easy to install, looks stable, reasonably expressive and convenient

Performance doesn't seem to scale well -- while ~15 tasks + 4 workers run under a second, 20 nodes up already gets over a minute.
But still within that minute, the optimal solution was actually found, just not declared optimal.

There are multiple options for further investment:
 - run the scheduler in parallel to actual computation, stopping early to obtain best-so-far,
 - run on immediate neighborhood of computable tasks,
 - use only as a benchmarker for assessing competitive factor of heuristic solutions.

A particular issue is that the solver does not penalize solutions with lot of data transfer, as long as the total runtime is optimal -- in other words, the model does not capture the noisy neigbor aspect of reality.
A similar deficiency would appear when solving only a subgraph -- we don't have a good number that captures the "fitness for further computations".
