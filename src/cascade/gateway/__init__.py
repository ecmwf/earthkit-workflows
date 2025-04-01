# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""This module is responsible for managing multiple cascade jobs:
    - spawning new (slurm) jobs, presumably based on some frontend instructions,
    - monitoring their being alive,
    - receiving progress updates and data results from the jobs,
    - serving progress/result queries, presumably from a frontend.

It is a standalone single process, with multiple zmq sockets (large data requests, regular request-response, update stream, ...).
"""
