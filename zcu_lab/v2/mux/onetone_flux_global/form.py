"""Captured mux onetone and global DC form defaults."""

from functools import partial

from pydantic import JsonValue
from zcu_tools.cfg_model import FieldMeta, Range, Slot
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2.support.measure.flux_defaults import captured_flux_seed
from zcu_lab.v2.support.measure.mux_bindings import require_mux_group
from zcu_lab.v2.support.measure.mux_flux_forms import global_flux_interval
from zcu_lab.v2.support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_names,
    captured_mux_participants,
    mux_readout_initial,
    mux_readout_seed_metadata,
)
from zcu_lab.v2.support.measure.seed_metadata import seed_metadata

from .core import OnetoneFluxGlobalCfg


def _initial_values(
    env: FormEnv, defaults: dict[str, JsonValue], *, names: tuple[str, ...]
) -> dict[str, JsonValue]:
    group = require_mux_group(env.parameters)
    seeds = [captured_flux_seed(env, name, group.flux_dev) for name in names]
    units = {seed.unit for seed in seeds if seed.unit is not None}
    if len(units) > 1:
        raise ValueError("Mux flux devices must share one native unit")
    start, stop = global_flux_interval(
        [None if seed.points is None else seed.points.integer for seed in seeds],
        [None if seed.points is None else seed.points.half for seed in seeds],
    )
    defaults.update(
        {
            "flux_dev": group.flux_dev,
            "flux_unit": next(iter(units)) if units else None,
            "flux": {"kind": "linrange", "start": start, "stop": stop, "n": 101},
        }
    )
    return defaults


def make_form(ctx: FormContext) -> Form[OnetoneFluxGlobalCfg]:
    """Capture full tone identities, ringup readout and a shared coil sweep.

    ctx supplies fixed Open/Reset group, aliases, library and device values.
    Seed global DC bounds from the union of all captured Q half/integer ranges;
    missing calibration leaves endpoints blank, invalid sources raise. df starts
    at -1..1 nominal MHz with 101 points. Participants selects acquisition order,
    not the tone picker. Candidates stay editable and are not calibrations.
    """
    names = captured_mux_names(ctx)
    participants = captured_mux_participants(ctx, names)
    readout = mux_readout_initial(ctx, wait_for_ringup=True)
    metadata = mux_readout_seed_metadata(readout)
    metadata[("readout",)] = FieldMeta(
        description="Captured $mux.module.readout slot, or readout/mux:blank from group wiring; confirm calibration"
    )
    metadata[("flux_dev",)] = FieldMeta(
        description="Device name captured from $mux.flux_dev"
    )
    for endpoint in ("start", "stop"):
        metadata[("flux", endpoint)] = seed_metadata(
            "Union of actual half/integer intervals from each qubit.flux on $mux.flux_dev, accounting for captured other device biases; native device unit"
        )
    metadata[("flux", "n")] = seed_metadata(
        "Shared 101-point global DC scan candidate", fallback=True
    )
    for endpoint in ("start", "stop", "n"):
        metadata[("df", endpoint)] = seed_metadata(
            "Nominal common offset candidate -1..1 MHz, 101 points", fallback=True
        )
    for field, description in (
        ("reps", "Fixed 1000 repetitions per round candidate"),
        ("rounds", "Fixed 10 software rounds candidate"),
        ("relax_delay", "Fixed 1.0 us recovery delay candidate"),
        ("initial_delay", "Fixed 1.0 us initial delay candidate"),
        ("settle", "Fixed 0.1 s DC settling candidate; confirm for each driver"),
    ):
        metadata[(field,)] = seed_metadata(description, fallback=True)
    return Form(
        OnetoneFluxGlobalCfg,
        tree={
            "reset": Slot("$mux.module.reset"),
            "readout": readout,
            "tones": [{"name": name} for name in names],
            "flux": Range(None, None, expts=101),
            "df": Range(-1.0, 1.0, expts=101),
            "participants": list(participants),
            "settle": 0.1,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": 1.0,
            "relax_delay": 1.0,
        },
        initial_values=partial(_initial_values, names=tuple(names)),
        fixed_elements=(
            FixedElements(("tones",), "name", ("participants",)),
            READOUT_ELEMENTS,
        ),
    ).with_seed_metadata(metadata)
