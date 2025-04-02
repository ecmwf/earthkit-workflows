# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A lightweight server for managing SharedMemory, ie, for passing output of one process as another's input.
Intended to run in every worker instance (regardless whether created by fiab, cascade-cluster, dask, ...)
Does not handle cross-worker coms on its own, but participates -- thats expected to work via
  1. controller issues command
  2. source worker gets the data from its shm and ships over to target worker
  3. target worker puts to its shm

Features:
 - uses python's SharedMemory -- no overhead, zero-copy sharing
 - persists to disk when running out of (adjustable) capacity, in an LRU-like fashion (see algorithms.py)
   - never overdraws its capacity -- a request that would do so is refused with a "wait" response
 - uses UDP server, since we are communicating locally only. Includes stateless client which hides this
   - *not* intended to be directly invoked by (remote) cascade controller -- the executor's worker handles this instead
 - it is safe wrt not paging out datasets currently being written or read by a task process. This however means
   that a livelock is possible, and task processes should not use infinite timeout on `get`/`allocate` calls
 - considers only memory occupation by its own datasets, ie, *not* global system memory occupation
   - however, its capacity can be dynamically adjusted. Intention is that the *worker* detects there is low system
     memory, and in turn decreses shm's capacity (to less than what /dev/shm would give), which causes a pageout

Usage:
 - at the start of the worker, launch the standalone `server` process
   - the server is single-threaded in its listening-serving functionality, but offloads all disk operations to
     a dedicated thread pool so that it never blocks -- client is returned a "wait" response and should try
     again later
     - # TODO implement the pool in .disk/.dataset
   - you must pass `port` parameter to the server,  and then `shm.client.ensure` to ensure the server is ready
     and serving. You need to call `shm.api.publish_client_port` in the process that will be spawning the worker
     processes
   - you may pass `capacity` parameter to the server, for maximum amount of shm it would create. If not given
     it defaults to the size of /dev/shm (which is an upper bound anyway). If overdraw would be imminent, it
     refuses said request and moves in the background something to disks until enough space available (if not possible,
     it refuses with "capacity exceeded" error message outright)
 - inside each task process in the worker, use the `allocate` and `get` calls in the client
   - the `allocate` call returns an object with a `memoryview` and `close` callback
   - the `get` call returns a `memoryview` and `close` callback
   - the `close` callbacks interact with the hidden SharedMemory object, and every time *must* be called, otherwise
     the server is left in inconsistent state
   - both `allocate` and `get` have a timeout param, to deal with the "wait" response
   - contextmanager for both is available to ensure proper .close() # TODO its not, impl in client.AllocatedBuffer
 - at the end of the worker's lifecycle, you may invoke `shm.client.shutdown` to free the /dev/shm and persisted data,
   or rely on sigterm handler do it for you
"""

# TODO it is worth considering bundling low-sized outputs (eg the "ok" no-output-placeholders) into a common page to
# reduce overhead (though the gains may be system dependent, possibly none)
