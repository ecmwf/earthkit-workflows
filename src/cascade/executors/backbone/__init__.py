"""
This module defines and implements communications backbone for multihost executors.
In particular, we can combine any local (that is, single-host) executor such as 
Fiab or dask.LocalCluster with any backbone such as http.starlette or zmq.

The role of backbone is simply to transmit commands from the controller (ActionSubmit,
ActionDatasetTransmit, ...) to the workers, and transmit Events back (TaskComplete,
DatasetAvailable, ...).

The `interface.py` revolves around three functionalities:
 - send_message & recv_messages -- used to send/recieve messages from a hardcoded list,
   with serde handled. Send is blocking only if the local queue is full (not expected given
   message processing should be fast), recv is blocking until first event, and returns
   all events available. In addition, there is send_message_callback, which provides
   a serializable means for executor-spawned processes to report state changes.
 - send_data & recv_data -- TODO
 - the bootstrap/registration is supposed to happen in the respective init calls, using the
   Register messages. The workers are expected to provide unique host/worker names already.
This interface is implemented by each individual backbone.


Then there are two adapter modules:
 - BackboneControllerExecutor -- a thin layer over the backbone that is given to the Controller.
   It implements the Executor interface by sending commands over the backbone to the individual workers.
 - BackboneLocalExecutor -- a thin layer that is given an Executor instance and handles communication between
   the backbone and it.

And one `serde` module
"""
