"""
Declares config and control classes
"""

from dataclasses import dataclass

@dataclass
class DaskDelayed:
    pass

@dataclass
class DaskThreaded:
    pass

@dataclass
class DaskFutures:
    workers: int
    dyn_sched: bool
    fusing: bool

@dataclass
class Fiab:
    dyn_sched: bool
    fusing: bool
    workers: int

Options = DaskDelayed|DaskThreaded|DaskFutures|Fiab
