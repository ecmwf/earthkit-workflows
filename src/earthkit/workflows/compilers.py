# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Provides compilers in various execution engines such as cascade. Intended
to be abstraction layer over those engines to prevent overly tight coupling.
Parallel to ekw.adapters -- but whereas adapters are imported by most of ekw,
compilers import from most of ekw.

Currently, only cascade is supported.
"""

import logging
from typing import Any, Literal

from earthkit.workflows.graph import Graph, serialise

try:
    from cascade.low.core import JobInstance
    from cascade.low.into import graph2job as cascadeInto
except ImportError:
    logger.error("failed to import any execution engine! Defaulting to dummy impl")
    def cascadeInto(graph: dict) -> Any:
        raise NotImplementedError("failed to import cascade execution engine")
    JobInstance = Any

Engine = Literal["cascade"]

def graph2job(graph: Graph, engine: Engine = "cascade") -> JobInstance:
    ser = serialise(graph)
    if engine != "cascade":
        raise NotImplementedError(f"not supported: {engine=}")
    return cascadeInto(ser)
