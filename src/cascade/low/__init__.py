# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Low Level representation of Cascade graphs -- not expected to be user facing.

Used to stabilise contract between Cascade graphs and Schedulers and Executors.

Works on atomic level: a single callable with all necessary information to be
executed in an isolated process.
"""

# TODO separate the scheduler/execution related concepts from core and schedule
# into a dedicated (sub)module
