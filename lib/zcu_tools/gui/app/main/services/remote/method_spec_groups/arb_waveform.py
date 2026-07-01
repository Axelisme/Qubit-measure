"""Arb Waveform remote method specs."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _bool_default,
    _expected_versions,
    _json,
    _str,
)

SPECS: dict[str, MethodSpec] = {
    "arb_waveform.list": MethodSpec(
        5.0,
        "List qubit-scoped arbitrary waveform data keys. Returns {waveforms: [name]}.",
        tool_name="list_arb_waveform",
    ),
    "arb_waveform.preview": MethodSpec(
        10.0,
        "Load one arbitrary waveform asset and render a normalized I/Q/Abs preview "
        "PNG. Returns {recipe, preview_figure}; recipe is null for raw imported "
        "assets.",
        (_str("name", "Arbitrary waveform data_key"),),
        tool_name="get_arb_waveform_preview",
    ),
    "arb_waveform.set": MethodSpec(
        10.0,
        "Create or overwrite an arbitrary waveform from a formula recipe. The recipe "
        "fully replaces waveform data and is embedded into the single .npz asset. "
        "Returns {success, status, preview_figure}.",
        (
            _str("name", "Arbitrary waveform data_key"),
            _json("recipe", "Formula recipe object"),
            _bool_default("overwrite", False, "Allow replacing an existing data_key"),
            _expected_versions(),
        ),
        tool_name="set_arb_waveform",
    ),
}
