"""
Implementation of schedulers.

The scheduler is intended as follows:
 - for a (static) JobInstance object, we create a graph decomposition which we call "schedule".
   This is independent of the environment -- we just determine a rough order in which tasks should
   be completed, and which allows parallel execution
 - the Schedule object is given to the Controller, which inside the `plan` action then proceeds
   through it, using the current state of the environment to issue commands to executor

In other words, the schedulers here are rather graph decomposition for efficient dynamic scheduling at runtime
"""

# TODO
# - change the Schedule structure from layer-linear to layer-tree
# - order within layers -- consider leaf weights, prioritize paths according to weight-freed
