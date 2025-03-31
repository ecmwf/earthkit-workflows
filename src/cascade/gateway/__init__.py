"""
This module is responsible for managing multiple cascade jobs:
    - spawning new (slurm) jobs, presumably based on some frontend instructions,
    - monitoring their being alive,
    - receiving progress updates and data results from the jobs,
    - serving progress/result queries, presumably from a frontend.

It is a standalone single process, with multiple zmq sockets (large data requests, regular request-response, update stream, ...).
"""
