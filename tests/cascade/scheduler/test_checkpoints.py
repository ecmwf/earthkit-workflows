from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.scheduler.precompute import precompute
from cascade.scheduler.checkpoints import trim_with_persisted
from cascade.low.core import JobInstanceRich, DatasetId

def test_trim_with_checkpoints():
    # we have a graph with
    # - 2 sources,
    # - 1 transform (combining the sources),
    # - 2 products (one consuming the transform, another consuming a source),
    # - and 1 sink (consuming both products).
    # We will checkpoint on the transform, meaning 1 source should be trimmed

    jobInstanceOrig = (
        JobBuilder()
            .with_node(
                "source1",
                TaskBuilder.from_entrypoint("whatever", {}, "Any"),
            )
            .with_node(
                "source2",
                TaskBuilder.from_entrypoint("whatever", {}, "Any"),
            )
            .with_node(
                "transform",
                TaskBuilder.from_entrypoint("whatever", {"i1": "Any", "i2": "Any"}, "Any"),
            )
            .with_edge("source1", "transform", "i1")
            .with_edge("source2", "transform", "i2")
            .with_node(
                "product1",
                TaskBuilder.from_entrypoint("whatever", {"i": "Any"}, "Any"),
            )
            .with_edge("transform", "product1", "i")
            .with_node(
                "product2",
                TaskBuilder.from_entrypoint("whatever", {"i": "Any"}, "Any"),
            )
            .with_edge("source2", "product2", "i")
            .with_node(
                "sink",
                TaskBuilder.from_entrypoint("whatever", {"i1": "Any", "i2": "Any"}, "Any"),
            )
            .with_edge("product1", "sink", "i1")
            .with_edge("product2", "sink", "i2")
    ).build().get_or_raise()
    jobRich = JobInstanceRich(jobInstance=jobInstanceOrig, checkpointSpec=None)
    preschedule = precompute(jobRich.jobInstance)
    persisted = {DatasetId(task="transform", output="0")}

    jobInstanceNew, preschedule, persisted_valid = trim_with_persisted(jobRich, preschedule, persisted)
    assert persisted_valid == persisted
    assert set(jobInstanceNew.tasks.keys()) == {'source2', 'transform', 'product1', 'product2', 'sink'}
