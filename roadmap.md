# Roadmap

# Improvements Pool
## Speed
* pre-schedule
  * persist the result, allow re-usal
  * improve algorithm, improve implementation
  * run in parallel to cluster spin-up
* assignments / schedule
  * fusing in case of single remaining output and good ratio of available data
  * opportunistic data transfers
  * allow partial component migration, allow pre-completion migration
  * proper treatment of long lived workers
  * allow worker idleness if no good assignment to be had
  * recompute instead of transmit // would probably need to be done together with some eager sends or broadcasts
  * don't always serialize & publish, multi-threading on worker // very questionable
  * re-frame the whole problem as sink partitioning and source-sink path tracing. Would lead to better fusing and better migrations. Migration should decompose the whole component in two part, if possible -- or not do it at all! It's not that much about prereqs (just broadcast all, or even recompute), but more about sinks. Or decompose the graph into source-sink paths (those with heaviest cost), and fuse this so that a single worker can handle this, and the other workers just compute opportunistically. Requires predictable hard path
* ipc
  * have worker2executor callback use an ipc socket instead of tcp
* serde
  * support for pickle-free serde
* profile/resource
  * make scheduler aware of task profiles in general, as well as transmit costs
## Stability
* long lived worker memory capacity / pressure, dropping of datasets
* eager partial Purge sent from controller
* re-tries for more messages than just register and data transmit
* recover from worker oom crash etc
* watermarks/checkpoints for job crashes and restarts
## Portability
* support k8s submit (in parallel to slurm submits)
## Functionality
* generator output
* metadata on outputs, for proper publishing, retrieval -- eg., `step_id` on ML model output
* single graph node running on multiple hosts // quite questionable, may be needed for some huge models
## Remodularization
* factor out the graph algorithms in scheduler, rewrite to eg rust
* consider [zict](https://zict.readthedocs.io/en/latest/) in place of shm / worker mem manager
  * rewrite shm to eg rust, make a standalone lib
* ray core / ray compiled graphs as a wholesome alternative
* [torch distributed](https://pytorch.org/docs/stable/distributed.html#tcp-initialization) for the data dissemination
