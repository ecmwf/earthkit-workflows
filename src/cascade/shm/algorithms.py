# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pageout algorithms -- selection of which datasets to send from shm to disk

Logic:
 - first those that have been accessed *exactly* once, ordered by created asc
   - rationale: consumed and wont be accessed again
 - then those that have been accessed *more times*, ordered by last access asc
   - rationale: a fan-out dataset, may be accessed again if still hot
 - lastly those that havent been accessed yet, ordered by created desc
   - rationale: already in trouble, lets make the older parts of the tree finish first
"""

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    key: str
    created: int
    retrieved_first: int
    retrieved_last: int
    size: int


def lottery(entities: Iterable[Entity], amount: int) -> list[str]:
    freed = 0
    consumedOnce = []
    consumedMult = []
    consumedNevr = []
    for entity in entities:
        if entity.retrieved_first == 0:
            consumedNevr.append(entity)
        elif entity.retrieved_first == entity.retrieved_last:
            consumedOnce.append(entity)
        else:
            consumedMult.append(entity)

    winners = []
    consumedOnce = sorted(consumedOnce, key=lambda e: e.created)
    for e in consumedOnce:
        winners.append(e.key)
        freed += e.size
        if freed >= amount:
            return winners

    consumedMult = sorted(consumedMult, key=lambda e: e.retrieved_last)
    for e in consumedMult:
        winners.append(e.key)
        freed += e.size
        if freed >= amount:
            return winners

    # NOTE this also includes stale_create. But stale_read are purged earlier
    consumedNevr = sorted(consumedNevr, key=lambda e: -e.created)
    for e in consumedNevr:
        winners.append(e.key)
        freed += e.size
        if freed >= amount:
            return winners

    # we don't really need to indicate whether it was succesfull or not
    return winners
