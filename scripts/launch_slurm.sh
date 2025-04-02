#!/bin/bash

# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $1

srun -J cascade-devel-01 --nodes=$((EXECUTOR_HOSTS+1)) --ntasks-per-node=1 --qos=np $SCRIPT_DIR/slurm_entrypoint.sh $1
