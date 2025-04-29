# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools

import numpy as np
import pytest

from earthkit.workflows.fluent import Payload, from_source


@pytest.fixture(scope="function")
def task_graph(request):
    func = getattr(request, "param", functools.partial(np.random.rand, 2, 3))
    return (
        from_source(
            [
                np.fromiter(
                    [func for _ in range(6)],
                    dtype=object,
                )
                for _ in range(7)
            ],
            dims=["x", "y"],
        )
        .mean("x")
        .min("y")
        .expand("z", internal_dim=1, dim_size=3, axis=0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )
