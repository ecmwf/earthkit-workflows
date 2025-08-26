# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Intended to test that various 3rd-party callables work under cascade

Similar modules will be found among earthkit-workflows plugins, all utilizing the
cascade.benchmarks.tests module
"""

from cascade.benchmarks.tests import CallableInstance, run_test


def test_numpy():

    def myfunc(l: int) -> float:
        import numpy as np

        return np.arange(l).sum()

    ci = CallableInstance(
        func=myfunc, kwargs={"l": 4}, args=[], env=["numpy"], exp_output=6
    )

    run_test(ci, "numpyTest1", 2)
