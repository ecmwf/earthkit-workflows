# from dask.distributed.scheduler import Scheduler

# """
# LocalCluster and KubeCluster are passed in as arguments to dask.distributed Client so
# we just need to override the functions in there where the scheduler class is defined

# For local cluster, Scheduler class is used in distributed/distributed/deploy/local.py
# For kubenertes cluster Scheduler class is specified in KubeCluster class in
# scheduler_spec dask-kubernetes/dask_kubernetes/classic
# /kubecluster.py


# NEED to handle forgetting completed jobs!

# """

# class StaticScheduler(Scheduler):


# def _transition_waiting_processing(self, key: Key, stimulus_id: str) -> RecsMsgs:

# def decide_worker_rootish_queuing_disabled(
#         self, ts: TaskState
#     ) -> WorkerState | None:
#         """Pick a worker for a runnable root-ish task, without queuing.

#         This attempts to schedule sibling tasks on the same worker, reducing future data
#         transfer. It does not consider the location of dependencies, since they'll end
#         up on every worker anyway.

#         It assumes it's being called on a batch of tasks in priority order, and
#         maintains state in `SchedulerState.last_root_worker` and
#         `SchedulerState.last_root_worker_tasks_left` to achieve this.

#         This will send every runnable task to a worker, often causing root task
#         overproduction.

#         Returns
#         -------
#         ws: WorkerState | None
#             The worker to assign the task to. If there are no workers in the cluster,
#             returns None, in which case the task should be transitioned to
#             ``no-worker``.
#         """
#         if self.validate:
#             # See root-ish-ness note below in `decide_worker_rootish_queuing_enabled`
#             assert math.isinf(self.WORKER_SATURATION)

#         pool = self.idle.values() if self.idle else self.running
#         if not pool:
#             return None

#         tg = ts.group
#         lws = tg.last_worker
#         if (
#             lws
#             and tg.last_worker_tasks_left
#             and lws.status == Status.running
#             and self.workers.get(lws.address) is lws
#         ):
#             ws = lws
#         else:
#             # Last-used worker is full, unknown, retiring, or paused;
#             # pick a new worker for the next few tasks
#             ws = min(pool, key=partial(self.worker_objective, ts))
#             tg.last_worker_tasks_left = math.floor(
#                 (len(tg) / self.total_nthreads) * ws.nthreads
#             )

#         # Record `last_worker`, or clear it on the final task
#         tg.last_worker = (
#             ws if tg.states["released"] + tg.states["waiting"] > 1 else None
#         )
#         tg.last_worker_tasks_left -= 1

#         if self.validate and ws is not None:
#             assert self.workers.get(ws.address) is ws
#             assert ws in self.running, (ws, self.running)

#         return ws

#     def decide_worker_rootish_queuing_enabled(self) -> WorkerState | None:
#         """Pick a worker for a runnable root-ish task, if not all are busy.

#         Picks the least-busy worker out of the ``idle`` workers (idle workers have fewer
#         tasks running than threads, as set by ``distributed.scheduler.worker-saturation``).
#         It does not consider the location of dependencies, since they'll end up on every
#         worker anyway.

#         If all workers are full, returns None, meaning the task should transition to
#         ``queued``. The scheduler will wait to send it to a worker until a thread opens
#         up. This ensures that downstream tasks always run before new root tasks are
#         started.

#         This does not try to schedule sibling tasks on the same worker; in fact, it
#         usually does the opposite. Even though this increases subsequent data transfer,
#         it typically reduces overall memory use by eliminating root task overproduction.

#         Returns
#         -------
#         ws: WorkerState | None
#             The worker to assign the task to. If there are no idle workers, returns
#             None, in which case the task should be transitioned to ``queued``.

#         """
#         if self.validate:
#             # We don't `assert self.is_rootish(ts)` here, because that check is
#             # dependent on cluster size. It's possible a task looked root-ish when it
#             # was queued, but the cluster has since scaled up and it no longer does when
#             # coming out of the queue. If `is_rootish` changes to a static definition,
#             # then add that assertion here (and actually pass in the task).
#             assert not math.isinf(self.WORKER_SATURATION)

#         if not self.idle_task_count:
#             # All workers busy? Task gets/stays queued.
#             return None

#         # Just pick the least busy worker.
#         # NOTE: this will lead to worst-case scheduling with regards to co-assignment.
#         ws = min(
#             self.idle_task_count,
#             key=lambda ws: len(ws.processing) / ws.nthreads,
#         )
#         if self.validate:
#             assert self.workers.get(ws.address) is ws
#             assert not _worker_full(ws, self.WORKER_SATURATION), (
#                 ws,
#                 _task_slots_available(ws, self.WORKER_SATURATION),
#             )
#             assert ws in self.running, (ws, self.running)

#         return ws

#     def decide_worker_non_rootish(self, ts: TaskState) -> WorkerState | None:
#         """Pick a worker for a runnable non-root task, considering dependencies and
#         restrictions.

#         Out of eligible workers holding dependencies of ``ts``, selects the worker
#         where, considering worker backlog and data-transfer costs, the task is
#         estimated to start running the soonest.

#         Returns
#         -------
#         ws: WorkerState | None
#             The worker to assign the task to. If no workers satisfy the restrictions of
#             ``ts`` or there are no running workers, returns None, in which case the task
#             should be transitioned to ``no-worker``.
#         """
#         if not self.running:
#             return None

#         valid_workers = self.valid_workers(ts)
#         if valid_workers is None and len(self.running) < len(self.workers):
#             # If there were no restrictions, `valid_workers()` didn't subset by
#             # `running`.
#             valid_workers = self.running

#         if ts.dependencies or valid_workers is not None:
#             ws = decide_worker(
#                 ts,
#                 self.running,
#                 valid_workers,
#                 partial(self.worker_objective, ts),
#             )
#         else:
#             # TODO if `is_rootish` would always return True for tasks without
#             # dependencies, we could remove all this logic. The rootish assignment logic
#             # would behave more or less the same as this, maybe without guaranteed
#             # round-robin though? This path is only reachable when `ts` doesn't have
#             # dependencies, but its group is also smaller than the cluster.

#             # Fastpath when there are no related tasks or restrictions
#             worker_pool = self.idle or self.workers
#             # FIXME idle and workers are SortedDict's declared as dicts
#             #       because sortedcontainers is not annotated
#             wp_vals = cast("Sequence[WorkerState]", worker_pool.values())
#             n_workers = len(wp_vals)
#             if n_workers < 20:  # smart but linear in small case
#                 ws = min(wp_vals, key=operator.attrgetter("occupancy"))
#                 assert ws
#                 if ws.occupancy == 0:
#                     # special case to use round-robin; linear search
#                     # for next worker with zero occupancy (or just
#                     # land back where we started).
#                     start = self.n_tasks % n_workers
#                     for i in range(n_workers):
#                         wp_i = wp_vals[(i + start) % n_workers]
#                         if wp_i.occupancy == 0:
#                             ws = wp_i
#                             break
#             else:  # dumb but fast in large case
#                 ws = wp_vals[self.n_tasks % n_workers]

#         if self.validate and ws is not None:
#             assert self.workers.get(ws.address) is ws
#             assert ws in self.running, (ws, self.running)

#         return ws


# # def decide_worker(
# #     ts: TaskState,
# #     all_workers: set[WorkerState],
# #     valid_workers: set[WorkerState] | None,
# #     objective: Callable[[WorkerState], Any],
# # ) -> WorkerState | None:
