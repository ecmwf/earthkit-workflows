# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from earthkit.workflows.fluent import Action, from_source

DataCube = dict[str, Any]

def mock_action(datacubes: DataCube | list[DataCube]) -> Action:
    if not isinstance(datacubes, list):
        datacubes = [datacubes]
    return from_source("test", datacubes=datacubes)
