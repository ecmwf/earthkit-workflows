"""Demonstrates gang scheduling capabilities, ie, multiple nodes capable of mutual communication.

The job is a source -> (dist group) -> sink, where:
    source just returns an int,
    dist group is L nodes to be scheduled as a single gang
        rank=0 node broadcasts a buffer containing the node's input
        each node returns its input multiplied by broadcasted buffer
    sink returns the sum of all inputs

There are multiple implementations of that:
    torch
    jax (actually does a mesh-shard global sum instead of broadcast -- the point is to showcase dist init)
"""

import os

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance, SchedulingConstraint


def source_func() -> int:
    return 42


def dist_func_torch(a: int) -> int:
    import datetime as dt

    import numpy as np
    import torch.distributed as dist

    world_size = int(os.environ["CASCADE_GANG_WORLD_SIZE"])
    rank = int(os.environ["CASCADE_GANG_RANK"])
    coordinator = os.environ["CASCADE_GANG_COORDINATOR"]
    print(f"starting with envvars: {rank=}/{world_size=}, {coordinator=}")
    dist.init_process_group(
        backend="gloo",
        init_method=coordinator,
        timeout=dt.timedelta(minutes=1),
        world_size=world_size,
        rank=rank,
    )
    group_ranks = np.arange(world_size, dtype=int)
    group = dist.new_group(group_ranks)

    if rank == 0:
        buf = [a]
        dist.broadcast_object_list(buf, src=0, group=group)
        print("broadcast ok")
    else:
        buf = np.array([0], dtype=np.uint64)
        dist.broadcast_object_list(buf, src=0, group=group)
        print(f"broadcast recevied {buf}")

    return a * buf[0]


def dist_func_jax(a: int) -> int:
    world_size = int(os.environ["CASCADE_GANG_WORLD_SIZE"])
    rank = int(os.environ["CASCADE_GANG_RANK"])
    coordinator = os.environ["CASCADE_GANG_COORDINATOR"]
    os.environ["JAX_NUM_CPU_DEVICES"] = "1"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jp

    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_platform_name", "cpu")
    # NOTE neither of the above seems to actually help with an init error message :(
    print(f"starting with envvars: {rank=}/{world_size=}, {coordinator=}")
    if coordinator.startswith("tcp://"):
        coordinator = coordinator[len("tcp://") :]
    jax.distributed.initialize(coordinator, num_processes=world_size, process_id=rank)
    assert jax.device_count() == world_size

    mesh = jax.make_mesh((world_size,), ("i",))
    global_data = jp.arange(world_size)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("i"))
    global_array = jax.device_put(global_data, sharding)
    result = jp.sum(global_array)
    print(f"worker {rank}# got result {result=}")
    return a + result


def build_dist_func(impl: str):
    if impl == "torch":
        return dist_func_torch
    elif impl == "jax":
        return dist_func_jax
    else:
        raise NotImplementedError(impl)


def sink_func(**kwargs) -> int:
    c = 0
    for _, v in kwargs.items():
        c += v
    print(f"sink accumulated {c}")
    return c


def get_job() -> JobInstance:
    source_node = TaskBuilder.from_callable(source_func)
    sink_node = TaskBuilder.from_callable(sink_func)
    job = JobBuilder().with_node("source", source_node).with_node("sink", sink_node)
    L = int(os.environ["DIST_L"])
    IMPL = os.environ["DIST_IMPL"]
    node = TaskBuilder.from_callable(build_dist_func(IMPL))

    for i in range(L):
        job = (
            job.with_node(f"proc{i}", node)
            .with_edge("source", f"proc{i}", "a")
            .with_edge(f"proc{i}", "sink", f"v{i}")
        )
        job.nodes["sink"].definition.input_schema[
            f"v{i}"
        ] = "int"  # TODO put some allow_kw into TaskDefinition instead to allow this

    job = job.build().get_or_raise()
    job.ext_outputs = list(job.outputs_of("sink"))
    job.constraints = [SchedulingConstraint(gang=[f"proc{i}" for i in range(L)])]
    return job
