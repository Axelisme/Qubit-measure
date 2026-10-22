"""Pulse-readout initial values and captured analysis names for a wide survey."""

from math import isfinite

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone._form_shared import initial_readout_timing

from .core import RangeFindCfg


def initial_resonators(ctx: FormContext) -> list[str]:
    """Seed editable names from the captured mux group, or empty without a mux.

    All members with present, finite MHz frequencies are sorted by frequency.
    Any missing/null frequency preserves the entire group order. Malformed
    group names or set nonfinite/nonnumeric frequencies raise ValueError.
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
    if not isinstance(raw, list) or not raw:
        raise ValueError("$mux.resonators must be a nonempty list of names")
    names: list[str] = []
    for name in raw:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("$mux.resonators must contain nonblank strings")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("$mux.resonators must contain unique names")
    frequencies: dict[str, float] = {}
    for name in names:
        source = parameters.lookup_point(f"{name}.freq")
        value = source.value
        if not source.present or value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(f"{name}.freq must be finite MHz to seed resonator order")
        frequencies[name] = float(value)
    if len(frequencies) != len(names):
        return names
    return sorted(names, key=frequencies.__getitem__)


def make_form(ctx: FormContext) -> Form[RangeFindCfg]:
    """Use freq linear's editable defaults plus a persisted analysis name list.

    ctx is the Open/Reset capture. The operator widens the absolute MHz sweep;
    names are editable and do not select hardware channels. Timing failures
    propagate; no hardware I/O or live parameter refresh occurs here.
    """
    return Form(
        RangeFindCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    set={
                        "freq": Hidden(0.0),
                        "gain": 0.05,
                        **initial_readout_timing(ctx),
                    },
                ),
            },
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=301
                )
            },
            "relax_delay": 1.0,
            "reps": 100,
            "rounds": 100,
            "initial_delay": Hidden(),
            "resonators": [name for name in initial_resonators(ctx)],
        },
    )
