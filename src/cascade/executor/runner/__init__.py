"""
This module is responsible for running actual tasks -- a thin wrapper over a Callable that handles:
 - getting & deserializing the inputs
 - serializing & storing the outputs
 - invoking callback to report task success, dataset publication, failures
 - setting up the environment: packages and envvars

The runner is a long-lived process, spawned by the executor module, with TaskSequence command
being sent to it over ipc zmq.

The submodules are:
 - packages -- a context manager for venv extension
 - memory -- a context manager for handling input & output datasets, and preserving values across
   jobs in TaskSequence(s)
 - runner -- the callable invocation itself
 - entrypoint -- the process's start which listens on zmq and invokes runner for each TaskSequence
"""

# NOTE there are a few performance optimisations at hand:
# - we could start obtaining inputs while we are venv-installing (but beware deser needing venv!)
# - we could start venv-installing & inputs-obtaining while previous task is running
# - we could be inputs-obtaining in parallel
# Ideally, all deser would be doable outside Python -- then this whole module could be eg rust & threads
