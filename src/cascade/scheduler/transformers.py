from cascade.low.core import JobExecutionRecord, Schedule

# TODO implement something reasonable here -- like safe transformer, with max size, mem limits, ...

class FusingTransformer():
    def transform(self, schedule: Schedule, record: JobExecutionRecord|None = None) -> Schedule:
        # fuses everything assigned to a given host into a single task
        return Schedule(
            host_task_queues={
                host: [[ task for subgraph in subgraphs for task in subgraph ]]
                for host, subgraphs in schedule.host_task_queues.items()
            }
        )
