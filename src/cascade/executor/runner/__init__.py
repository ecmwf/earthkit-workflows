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
 - entrypoint -- the callable invocation itself
"""
