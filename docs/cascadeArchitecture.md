# Cascade system architecture (brief)

Cascade is a distributed workflow runtime built around three stable execution layers:

1. **Scheduler** decides what can run next and where.
2. **Controller** drives the runtime loop and turns scheduling decisions into execution commands.
3. **Executor** runs those commands on each host and manages worker processes plus host-local data movement.

Around these layers, **Gateway** provides multi-job lifecycle management and an external API.

## Main runtime roles

| Role | Responsibility | Typical process count |
| --- | --- | --- |
| Gateway | Accept job submissions, start jobs, expose progress/results API | 1 per deployment/control plane |
| Controller | Per-job orchestration loop (schedule, dispatch, react to events) | 1 per job |
| Executor | Per-host runtime agent (worker lifecycle, data server, shared memory coordination) | 1 per host in the job |
| Worker (Runner) | Execute task sequences in isolated worker processes | Many per executor/host |
| Bridge | Controller-side communication adapter to all executors | In-controller component |

## Who launches what

At a high level, startup follows this chain:

`Gateway -> job process -> Controller + Executors -> Workers`

- **Gateway** can start jobs locally, via Slurm wrappers, or via an external launcher path.
- A **job process** runs the controller-side orchestration and starts/joins executor processes depending on deployment mode.
- Each **Executor** starts host-local services (notably a data server and shared-memory service), then starts worker runner processes.
- **Workers** execute task sequences assigned by the controller through the executor.

## Communication model (interrelations)

Cascade is message-driven and uses asynchronous channels between roles:

- **Gateway <-> clients**: request/response API for submit, progress, result retrieval, cleanup, shutdown.
- **Controller -> Executors**: control commands (task sequences, purge, transmit/persist/retrieve instructions via bridge routing).
- **Executors -> Controller**: events and health/failure messages (dataset published, transfer confirmations, task/executor failures, lifecycle events).
- **Controller -> Gateway**: job progress and output publication reports (when gateway-managed submission is used).
- **Executor internal**:
  - executor main loop coordinates workers and control-plane messages,
  - data server handles bulk dataset transport/persist/retrieve paths,
  - workers execute callables and publish outputs/events back to executor/controller flow.

A practical separation is:

- **Control plane**: scheduler/controller decisions, task dispatch, lifecycle and health signaling.
- **Data plane**: dataset publication, transfer, retrieval, persistence, and host-local memory management.

## Execution loop (per job)

For one submitted job, the controller repeatedly:

1. reads environment and precomputed schedule context,
2. asks scheduler for next assignments,
3. dispatches assignments through the bridge to executors,
4. consumes executor events (task/data progress, completions, failures),
5. updates scheduling/execution state until outputs are complete or a failure terminates the run.

## Integration and deployment boundaries

For platform teams and integrators, the key boundaries are:

- **Gateway API boundary**: where external systems submit and observe jobs.
- **Job launch boundary**: where cluster-specific launchers (for example Slurm, Kubernetes controllers, or custom operators) can be attached.
- **Executor host boundary**: where host capabilities and policies (CPU/GPU/memory isolation, storage, network) are enforced.
- **Message contract boundary**: where custom components can interoperate if they honor controller/executor message semantics.

This structure keeps scheduling logic, orchestration logic, and host execution mechanics separated, while allowing different deployment orchestrators and integration surfaces around a stable core runtime model.
