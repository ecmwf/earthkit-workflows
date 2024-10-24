"""
This module is a gateway to execution: given a Job, a schedule (static/dynamic), and an executor,
it issues commands based on the schedule to the executor until the job finishes or fails.

It declares the actual executor interface, but executors themselves are implemented in other
modules (cascade.executors) or packages (fiab).

The module is organised as follows:
 - core defines data structures such as State, Event, Action
 - executor defines the Executor protocol
 - notify, plan and act are implementation modules
 - impl is an implementation of the Controller protocol, via bundling the notify, plan, and act
   into the controller loop. This is the job execution entrypoint
"""
