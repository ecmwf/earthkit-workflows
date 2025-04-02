# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from earthkit.workflows import backends


class CustomBackend:
    def write(arg):
        return arg

    def sum(arg1, arg2):
        return arg1 + arg2


def test_custom():
    backends.register(str, CustomBackend)

    # For graph merging, custom functions must be the
    # unique e.g. attribute must be the same each access
    func = backends.write
    func1 = backends.write
    assert func == func1

    assert backends.write("Helloworld!") == backends.sum("Hello", "world!")
