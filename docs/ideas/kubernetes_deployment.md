# Cascade on Kubernetes: deployment options

This note explores how to run `cascade` on Kubernetes, using the architecture in `docs/cascadeArchitecture.md` and current code behavior in `src/cascade`.

## 1. General considerations

### 1.1 Tweaks and corrections to existing codebase

#### Networking and messaging

- **Make all cross-container channels explicitly routable.** Current code mixes local and network transports:
  - Controller <-> Executor uses ZeroMQ over `tcp://...` (good for cross-pod).
  - Executor <-> Worker uses `ipc:///tmp/...` (local only, fine if workers stay in same container as executor).
  - SHM control currently defaults to local unix socket path (`/tmp/cascShmSock-*`) via `CASCADE_SHM_SOCKET` (local only, fine if SHM server and workers stay in one container).
- **Stop relying on random inbound controller-report ports on Gateway.** `gateway.router` binds a random `tcp://<host>:<port>` per job for controller reports. This is hard to expose safely via Kubernetes Services and NetworkPolicies.
  - Prefer a fixed report endpoint (single stable port), or invert direction (Gateway pulls from job status store / queue).
- **Harden Gateway <-> Controller reporting reliability.** `controller.report` has a TODO about retries/acks; currently report delivery is best-effort PUSH. For ephemeral controllers and persistent gateway, add delivery guarantees (ack protocol or durable queue).

#### Gateway API boundary

- **Decouple external API from raw ZeroMQ framing.** Current Gateway client/server use internal binary payloads over ZMQ REQ/REP (`clazz` + JSON payload bytes).
- If external exposure is needed, keep this protocol internal and add a stable API facade (HTTP/gRPC) rather than exposing ZMQ directly.
- If your current deployment uses a local unix socket for gateway access, this must be replaced by a network-reachable endpoint (or an adapter sidecar) before Kubernetes service exposure.

#### Logging propagation and retrieval

- **Logging propagation is already mostly cascade-wide** through `LoggingConfig` (`gateway -> job launcher -> controller/executor -> workers`), but there are gaps:
  - `main_dist` still uses default logging in one path (`TODO handle logging for dist scenario`).
  - Troika/Slurm spawn paths have TODOs for proper logging propagation.
- **Prefer stdout/stderr logging as Kubernetes default.** Current default already logs to stderr; this aligns with `kubectl logs` and cluster log collection.
- **Clarify file logging semantics in containers.** If `path_base` is used, define where logs live (emptyDir vs PVC), rotation behavior, and collection mechanism.

#### Runtime state durability

- Gateway currently keeps job progress/results in memory. A gateway restart loses that state.
- For persistent service mode, progress/results metadata should be backed by durable storage (DB/object store), not only in-process memory.

### 1.2 New code on top

- **Kubernetes-facing API layer** in front of Gateway (REST/gRPC):
  - AuthN/AuthZ, multi-tenant routing, request validation, versioned API.
  - Translate API calls into current internal Gateway requests.
- **Kubernetes packaging**:
  - Helm chart / Kustomize for Gateway, Controller, Worker templates, Services, NetworkPolicies, RBAC.
  - ConfigMaps/Secrets for runtime and logging config.
- **Observability integration**:
  - Pod labels/annotations including `job_id`, `role`, `host_id`, `worker_id`.
  - Prometheus metrics and tracing span correlation across Gateway/Controller/Executor.
- **Durable state services**:
  - Metadata DB (job lifecycle/progress/result pointers).
  - Object store for large result payloads beyond in-memory retention.

## 2. Option analysis

## 2.1 Option A: single container (Gateway + Controller + Executor + Workers) ("no-scaling")

### Tweaks and corrections to existing codebase

- Minimal code change path.
- Keep local transports as-is (`ipc://` worker sockets, local SHM socket files).
- Ensure Gateway bind URL is explicitly configurable for in-pod vs external access.
- Prefer `LoggingConfig.path_base = null` for stdout/stderr-first logging in Kubernetes.

### New code on top

- Single Deployment/Pod spec, plus Service only if external access needed.
- Optional API facade sidecar (HTTP -> internal Gateway protocol).
- Liveness/readiness probes and resource limits.

### Notes

- Easiest to ship, but no horizontal scalability and weaker fault isolation.

## 2.2 Option B: Gateway outside Kubernetes, each submitted job as one Kubernetes pod with Controller + Executor(s) + Workers ("multiple no-scaling")

### Tweaks and corrections to existing codebase

- Gateway must launch Kubernetes Jobs instead of local subprocess/slurm/troika for this mode.
- Report channel (`report_address`) must be routable from job pod to external Gateway.
- Add retry/ack or durable buffering for controller reports to handle transient network failures.
- Make loggingConfig propagation complete for this launcher path.

### New code on top

- New launcher implementation in Gateway ("k8s submitter"):
  - Build pod spec from `JobSpec` and infra policy.
  - Inject serialized job instance + logging config + report endpoint.
- Network integration:
  - External DNS/service exposure for Gateway report endpoint.
  - TLS and firewall rules from cluster egress to Gateway ingress.
- Optional external API facade remains useful.

### Notes

- Good isolation per job; each job can fail independently.
- Still no internal per-job horizontal scaling if each pod is self-contained.

## 2.3 Option C: static multi-container deployment (Controller pod + Worker/Executor pods), Gateway either in cluster or outside ("static scaling")

### Tweaks and corrections to existing codebase

- Standardize addressing assumptions:
  - Ensure executor registrations (`maddress`, `daddress`, `url_base`) resolve via Kubernetes DNS.
  - Use fixed container ports and clear service discovery patterns.
- Verify `platform.get_bindabble_self()` behavior in pods; prefer explicit bind host/IP where needed.
- Preserve locality assumptions:
  - Keep worker processes inside executor pod/container (because worker IPC and SHM are local-only).
- Improve startup coordination:
  - Controller currently waits for `expected_executors`; static replica counts map well, dynamic joins do not.

### New code on top

- Static manifests/Helm values:
  - One controller Deployment/StatefulSet and N executor Deployments/StatefulSets.
  - Headless Services for stable DNS if needed.
- Config/secret distribution for shared job config and credentials.
- Optional in-cluster Gateway deployment + Service.

### Notes

- Real horizontal scaling possible, but mostly pre-provisioned/static.
- Better resource utilization than option A/B, but higher operational complexity.

## 2.4 Option D: operator-driven service model ("gateway as a service" + dynamic Controller+N Worker topology) ("service scaling")

### Tweaks and corrections to existing codebase

- Define a stable control contract for "job runtime spec" independent of CLI-only spawning.
- Remove assumptions tied to random ports and process-local orchestration.
- Introduce durable job metadata/result tracking; in-memory Gateway state is not enough for service semantics.
- Strengthen controller report reliability and idempotency (duplicate-safe progress/result updates).

### New code on top

- Kubernetes Operator + CRDs:
  - `CascadeJob` custom resource with desired topology/resources.
  - Reconciler creates controller pod + executor pods and tracks lifecycle.
- Gateway service layer:
  - Accept requests, create/update CRDs, expose status/results API.
- Optional queue/bus for async status/result ingestion (Gateway <-> Controller decoupling).
- Autoscaling policies (HPA/KEDA/custom) for executor counts and queue depth.

### Notes

- Highest flexibility and best long-term SaaS-style architecture.
- Highest implementation and ops complexity.

## 2.5 Additional option E: hybrid static worker pool + dynamic controllers

### Tweaks and corrections to existing codebase

- Allow controller to target a reusable pool of executors instead of job-owned executors.
- Add stricter isolation controls for dataset lifetimes and purge guarantees between jobs.

### New code on top

- Long-lived executor pool deployment.
- Scheduler/allocator service mapping incoming jobs to pool capacity.
- Admission control and quota enforcement per tenant.

### Notes

- Lower startup latency than per-job spin-up, but isolation/security needs careful design.

## 3. Focus topics requested

## 3.1 Inter-container communication, especially ZeroMQ

### Tweaks and corrections to existing codebase

- Keep ZMQ over TCP for cross-pod links only (controller <-> executor, gateway <-> controller reports if retained).
- Keep IPC/unix socket links strictly inside a pod/container boundary (executor <-> workers, SHM internals).
- Replace per-job random Gateway report port binding with stable endpoint pattern.
- Add explicit timeouts/retries/acks for controller report channel.

### New code on top

- Kubernetes Services/headless Services for routable endpoints.
- NetworkPolicies allowing only required role-to-role flows.
- Optional message broker for reliable report ingestion instead of direct PUSH socketing.

## 3.2 Logging configuration and retrieval

### Tweaks and corrections to existing codebase

- Complete logging propagation in all launch paths (local, dist, slurm, troika, k8s).
- Keep `LoggingConfig` as the single contract; add fields only when globally needed (for example role labels or structured metadata fields).
- Prefer stdout/stderr mode in Kubernetes; use file mode only with explicit volume strategy.

### New code on top

- Cluster log aggregation (Fluent Bit / Vector / OpenTelemetry collector).
- Correlate logs by labels (`job_id`, role, host, worker) from pod metadata and/or structured logs.
- Optional sidecar if file-based logs must be shipped.

## 3.3 Gateway-controller messaging with persistent Gateway and ephemeral Controller

### Tweaks and corrections to existing codebase

- Make report writes reliable and idempotent:
  - Retries + ack, or a durable intermediate transport.
  - Sequence/timestamp handling robust to duplicates and reordering.
- Externalize job state from Gateway memory to durable store.
- Ensure controller shutdown/failure is always reflected as terminal job state, even during network faults.

### New code on top

- Persistent job state backend (SQL/kv + object storage for payloads).
- Reconciliation loop that marks stalled jobs failed/unknown after heartbeat/report timeout.
- Optional dead-letter handling for failed report deliveries.

## 4. Recommended implementation sequence

1. Start with **Option A** in Kubernetes for packaging/ops baseline.
2. Add **Gateway API facade** (HTTP/gRPC) and stable endpointing.
3. Move to **Option C** for static multi-pod scale once networking and logging are stable.
4. Evolve to **Option D** (operator/service) when multi-tenant dynamic scaling and durability are required.
