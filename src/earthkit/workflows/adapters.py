# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Provides adapters to various execution engines such as cascade. Intended
to be abstraction layer over those engines to prevent overly tight coupling.
Parallel to ekw.compilers -- but whereas compilers imports from most of ekw,
adapters is imported by most of ekw.

Currently, only cascade is supported.
"""

import logging

try:
    from cascade.low.core import DefaultTaskOutput
    DefaultNodeOutput = DefaultTaskOutput
except ImportError:
    logger.error("failed to import any execution engine! Defaulting to dummy impl")
    DefaultNodeOutput = "0"
