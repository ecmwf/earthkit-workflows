from dataclasses import dataclass

@dataclass(frozen=True, eq=True, slots=True)
class JobSpec:
    workers: int
    hosts: int = 1
