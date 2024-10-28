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

Options = DaskDelayed|DaskThreaded|DaskFutures|Fiab
