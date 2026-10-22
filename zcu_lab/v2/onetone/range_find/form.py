"""Pulse-readout initial values and captured analysis names for a wide survey."""

from collections.abc import Sequence
from math import isfinite

from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import ParameterContext

from zcu_lab.v2.onetone._form_shared import initial_spectrum_tree

from .core import RangeFindCfg


def initial_resonators(ctx: FormContext) -> list[str]:
    """Seed editable names from the captured mux group, or empty without a mux.

    All members with present, finite MHz frequencies are sorted by frequency.
    Any missing/null frequency preserves the entire group order. Malformed
    source types or set nonfinite/nonnumeric frequencies raise ValueError.
    Duplicate names remain editable; only Analyze requires unique names.
    No readout settings are taken from the mux and no source is changed.
    """
    parameters = ctx.parameters
    if (
        parameters is None
        or parameters.roles is None
        or "mux" not in parameters.roles.names
    ):
        return []
    raw = parameters.value("$mux.resonators")
    if not isinstance(raw, list):
        raise ValueError("$mux.resonators must be a list of strings")
    names: list[str] = []
    for name in raw:
        if not isinstance(name, str):
            raise ValueError("$mux.resonators must contain strings")
        names.append(name)
    frequencies = resonator_frequencies(parameters, names)
    if frequencies is None:
        return names
    return [
        name
        for _, name in sorted(
            zip(frequencies, names, strict=True), key=lambda item: item[0]
        )
    ]


def resonator_frequencies(
    parameters: ParameterContext, names: Sequence[str]
) -> list[float] | None:
    """Read captured MHz in name order, or None if any frequency is absent/null.

    parameters is the fixed Open/Reset or Run/Load capture. Every set frequency
    must be finite numeric MHz (not bool), even if another name is missing;
    invalid values raise ValueError and source errors propagate unchanged.
    No sorting, live lookup or source mutation occurs.
    """
    frequencies: list[float] = []
    missing_frequency = False
    for name in names:
        source = parameters.lookup_point(f"{name}.freq")
        value = source.value
        if not source.present or value is None:
            missing_frequency = True
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(f"{name}.freq must be finite MHz to seed resonator order")
        frequencies.append(float(value))
    return None if missing_frequency else frequencies


def make_form(ctx: FormContext) -> Form[RangeFindCfg]:
    """Use freq linear's editable defaults plus a persisted analysis name list.

    ctx is the Open/Reset capture. The operator widens the absolute MHz sweep;
    names are editable and do not select hardware channels. Timing failures
    propagate; no hardware I/O or live parameter refresh occurs here.
    """
    return Form(
        RangeFindCfg,
        tree={
            **initial_spectrum_tree(ctx),
            "resonators": [name for name in initial_resonators(ctx)],
        },
    )
