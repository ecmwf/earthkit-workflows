# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Scheduler module is responsible for determining task->worker assignment.
This happens in the main Controller loop, where Events are received from
workers, provided to the Scheduler which then determines the assignment.
The Controller then converts those to Commands and sends over to the Workers.
In the meantime, Scheduler updates and calibrates internal structures,
so that upon next batch of Events it can produce good assignments.

There are multiple auxiliary submodules:
 - graph: decomposition algorithms and distance functions
 - core: data structures for assignment and schedule representation
 - assign: fitness functions for determining good assignments

These are all used from the `api` module here, which provides the interface
for the Controller.
"""
