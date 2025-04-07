# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from earthkit.workflows import Cascade


def get_graph(lead_time, ensemble_members, CKPT=None, date="2024-12-02T00:00"):
    import anemoicascade as ac

    CKPT = (
        CKPT
        or "/lus/h2resw01/hpcperm/ecm0672/pub/anemoi-ckpt/inference-aifs-0.2.1-anemoi.ckpt"
    )

    model_action = ac.fluent.from_input(
        CKPT, "mars", date, lead_time=lead_time, ensemble_members=ensemble_members
    )
    result = model_action.mean(dim="ensemble_member")
    result = result.map(print)

    cascade_model = Cascade.from_actions([result.sel(param="2t")])

    cascade_model.visualise(
        "model_running.html", preset="blob", cdn_resources="in_line"
    )
    return cascade_model._graph
