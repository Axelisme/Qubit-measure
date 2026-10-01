"""Arb Waveform remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    default_boolean,
    required_json,
    required_string,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "arb_waveform.list",
        "arb_waveform:h_arb_waveform_list",
        MethodSpec(
            5.0,
            "List qubit-scoped arbitrary waveform data keys. Returns {waveforms: [name]}.",
        ),
        agent=AgentMethodPolicy(reveals=("arb_waveforms",)),
    ),
    method_entry(
        "arb_waveform.preview",
        "arb_waveform:h_arb_waveform_preview",
        MethodSpec(
            10.0,
            "Load one arbitrary waveform asset and render a normalized I/Q/Abs preview "
            "PNG. Returns {recipe, preview_figure}; recipe is null for raw imported "
            "assets.",
            (required_string("name", "Arbitrary waveform data_key"),),
        ),
        agent=AgentMethodPolicy(reveals=("arb_waveforms",)),
    ),
    method_entry(
        "arb_waveform.set",
        "arb_waveform:h_arb_waveform_set",
        MethodSpec(
            10.0,
            "Create or overwrite an arbitrary waveform from a formula recipe. The recipe "
            "fully replaces waveform data and is embedded into the single .npz asset. "
            "Returns {success, status}; saving does not render a preview. "
            "Call arb_waveform.preview separately when a PNG is needed.",
            (
                required_string("name", "Arbitrary waveform data_key"),
                required_json("recipe", "Formula recipe object"),
                default_boolean(
                    "overwrite",
                    default=False,
                    desc="Allow replacing an existing data_key",
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            guard_deps=("arb_waveforms",), refresh_after_write=True
        ),
    ),
)
