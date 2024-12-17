"""
Scheduler module is responsible for determining task->worker assignment.
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
