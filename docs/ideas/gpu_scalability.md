# Context
Thoughts about GPU utilization on a k8s cluster, and how it interplays with the current gateway-controller design.
Assumes basic k8s deployment is already supported, which is not true at the moment -- see [known obstacles](./kubernetes_deployment.md).

# Idea
I'm now thinking whether it would make sense to use some long running inference service like KServe, eg in the multi host k8s deployments.
So the anemoi-inference task would not be about "select a gpu worker, load model there, run inference, output" but instead "send a request to a service and await it"

It would have some advantages, such as lower latency, in case most people would use the same model / smaller set of models.

It would perhaps be better for resource allocation -- our current design requires allocation of workers for the whole duration of the job.
Say we had one gpu in total on the cluster but 20 cpus, and 7 people would submit a job at the same time.
In the current design, they would have to be completely serialized, meaning initial condition fetch of one job would have to wait for complete finish of another job.
But if we switched to the service approach, they can all start concurrently, truly bottlenecking on the service itself only (which is valid).

We would probably not oversubscribe -- the gateway could be "when an inference-needing job arrives, check whether inference service is running, and if not start it concurrently; and shut it down when idleness expected".

It would have multiple some tho:
1. It restrics opportunistic gpu usage.
Say a job is inference + products, and some products would speed up when on gpu but is not a must have.
In the current design, one gpu-enhanceable product stays on the same worker and benefits from gashme, but others fan out on cpu.
In the inference-service model, everything cpus.
2. It robs the scheduler of visibility -- it would not see what other schedulers are about to do, but they share a resource together.

Possibly, both could be overcome, at the cost of a substantial redesign, of gateway and controller fusing.
There would be one controller-scheduler for all currently ongoing jobs, ie, everything would be one dynamic job.
Then there would be no need for a dedicated inference service, but in an extension to already discussed idea of scheduler being venv-aware, it could basically flag a certain worker as "holds torch + model venv".
The scheduler could then dynamically decide whether to transfer an inference's result to another worker to do products there and continue with interence-themed task on that worker (keeping weights in memory), *or* evict the weights and keeping the output and continue with a product on the worker's GPU -- whatever is better for the remaining schedule.
The current scheduler can already operate on jobs decomposed into connectivity components, and while a single component is not assumed to change, implementing a new component addition while a job is already in progress would not be inconceivable.

There would be another advantage -- we would support fairness in a better way, utilize more, and we could support priority better.
Current design, when workers already allocated, cannot take them away, even if they are idle.
When redone, the scheduler would see all jobs and all workers at once, and it could dynamically move workers between jobs.
We already support moving workers between components, but we do it only when a component is fully scheduled -- that gives us utilization without any extra work.
Adding priority would require implementing a new signal which would then be incorporated into scheduler decision making.

# Conflict
This current design came with primarily Slurm in mind, which itself is batch oriented in its allocations.
The question is then, should we orient future development of Cascade to be primarily Slurm-friendly, or would we re-orient to k8s?
In principle, we can support both -- we would need to introduce a notion of "job group" which would be the scheduler/controller sharing primitive, and add something like `job_group_id` to the new job request for gateway, it would then know whether to spawn a new one (on Slurm) or attach to an existing one (on K8s).

_Addendum_: turns out, after spending 30 seconds googling, that Slurm can dynamically resize jobs as well, hence the new Gateway-Controller would own all jobs at once, and have deployment-specific implementation of the "addComponent", "requestNode", "freeNode" API. And after a few minutes of more thinking, I think we should leave the "slurm job per cascade job" in favour of "slurm job per single worker", and have cascade jobs communicate across slurm jobs.
