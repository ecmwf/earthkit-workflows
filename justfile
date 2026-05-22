# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# this makes all the commands here use the local venv
set dotenv-path := ".env"

val:
    uv run ty check src
    uv run ty check tests
    uv run ty check integration_tests
    uv run pytest -n8 tests
fmt:
    uv run prek --all-files
integration testname:
    # testname is the importible module, so eg job_ekwTrivial
    cd integration_tests && uv run python harness.py {{testname}}
