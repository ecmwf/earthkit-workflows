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

@dataclass
class Fiab:
    workers: int

@dataclass
class MultiHost:
    workers_per_host: int
    hosts: int
    
@dataclass
class ZmqBackbone:
    workers_per_host: int
    hosts: int

Options = DaskDelayed|DaskThreaded|DaskFutures|Fiab|MultiHost|ZmqBackbone
