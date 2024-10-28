"""
Provides a unifying interface to a set of hosts that implement the controller.executor interface.
It is a "router/proxy" because it doesn't do any advanced logic, just routes and forwards requests
to remote executors via selected network interface.

Notes:
 - set of hosts has to be specified in advance and may not change during (in particular, connection loss
   is fatal). Will be improved later.
 - based on rest/http communication. Will be improved later.
 - assumes all hosts can talk to each other, and additionally implement the `transmit_to_url` and
   `receive_from_url` calls. This is not expected to change.
 - suboptimal with respect to latency -- events at each host are not gathered as they happened, but
   instead whenver the controller inquires. This will be later changed to hosts reporting their events
   as they happen for this router/proxy to store it in its queue, and controller's inquiries first
   retrieved from there.

The submodules:
 - `impl` is the `controller.executor` implentation itself
 - `worker_server` is a simple server to be executed at each host, to comm between `impl` here and `executor`
    of the host. It only relays commands to the `executor` or `shm.client`
 - `client` handles the calls issued by `impl`, ie, sending requests to `worker_server`s
 - `event_queue` facilitates somehow-async comms between this router/proxy and the workers: upon a `wait` call,
   we submit a request for each worker to a thread pool where they block and return the value into this queue
   (in the `client`). The router/proxy reads from this queue -- blocking until first result, then consuming
   as much in a non-blocking fashion as possible (to presumably gather results from previous call).
"""
