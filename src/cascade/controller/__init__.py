# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""This module is a gateway to execution: given a Job, a schedule (static/dynamic), and an executor,
it issues commands based on the schedule to the executor until the job finishes or fails.

It declares the actual executor interface, but executors themselves are implemented in other
modules (cascade.executors) or packages (fiab).

The module is organised as follows:
 - core defines data structures such as State, Event, Action
 - executor defines the Executor protocol
 - notify, plan and act are implementation modules
 - impl is an implementation of the Controller protocol, via bundling the notify, plan, and act
   into the controller loop. This is the job execution entrypoint

There are three auxiliary modules: views, dynamic and assignments, with not a great boundary.
Especially the latter two would best belong to the `cascade.scheduler` module instead, in some
organised fashion.
"""
