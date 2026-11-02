"""Captured mux twotone and global DC form defaults."""

from functools import partial

from pydantic import JsonValue
from zcu_tools.cfg_model import Expr, FieldMeta, FormNode, Hidden, Range, Slot, Template
from zcu_tools.gui.form import FormEnv, FormWritePath
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2.support.measure.flux_defaults import captured_flux_seed
from zcu_lab.v2.support.measure.mux_bindings import require_mux_group
from zcu_lab.v2.support.measure.mux_flux_forms import global_flux_interval
from zcu_lab.v2.support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_names,
    captured_mux_participants,
    mux_readout_initial,
    mux_tone_initials,
)
from zcu_lab.v2.support.measure.seed_metadata import seed_metadata

from .core import TwotoneFluxGlobalCfg


def make_form(ctx: FormContext) -> Form[TwotoneFluxGlobalCfg]:
    """Capture full tone drives/ranges, shared coil scan and readout defaults.

    ctx supplies group identities and calibration at Open/Reset. One tone picker
    edits original positions; participants alone selects acquisition order.
    Missing calibration leaves required bounds blank. The coil uses the union
    of actual half/integer intervals in its native unit. Invalid group mappings,
    unavailable DC infos and invalid calibration contributors raise. Refresh
    retains input and seed provenance; Reset captures sources again.
    """
    names = captured_mux_names(ctx)
    participants = captured_mux_participants(ctx, names)
    tones: list[dict[str, FormNode]] = [
        {
            "name": name,
            "drive": Template(
                "qub_probe",
                bindings={"qubit": name},
                # Run supplies this independent frequency scan.
                set={"freq": Hidden(0.0), "gain": 0.1, "waveform.length": 1.0},
            ),
            "freq": {
                "start": Expr(f"{name}.freq - 1.5 * {name}.kappa"),
                "stop": Expr(f"{name}.freq + 1.5 * {name}.kappa"),
            },
        }
        for name in names
    ]
    metadata: dict[FormWritePath, FieldMeta] = {
        ("readout",): FieldMeta(
            description="Captured $mux.module.readout slot, or readout/mux:blank from group wiring; confirm calibration"
        ),
        ("flux_dev",): FieldMeta(description="Device name captured from $mux.flux_dev"),
        ("flux", "n"): seed_metadata(
            "Shared 101-point global DC scan candidate", fallback=True
        ),
        ("expts",): seed_metadata(
            "Shared 301-point hardware frequency scan candidate", fallback=True
        ),
    }
    for field, description in (
        ("reps", "Fixed 1000 repetitions per round candidate"),
        ("rounds", "Fixed 10 software rounds candidate"),
        ("relax_delay", "Fixed 1.0 us recovery delay candidate"),
        ("initial_delay", "Fixed 1.0 us initial delay candidate"),
        ("settle", "Fixed 0.1 s DC settling candidate; confirm for each driver"),
    ):
        metadata[(field,)] = seed_metadata(description, fallback=True)
    for endpoint in ("start", "stop"):
        metadata[("flux", endpoint)] = seed_metadata(
            "Union of actual half/integer intervals from each qubit.flux on $mux.flux_dev, accounting for captured other device biases; native device unit"
        )
    for index, name in enumerate(names):
        for endpoint in ("start", "stop"):
            metadata[("tones", index, "freq", endpoint)] = seed_metadata(
                f"From captured {name}.freq and {name}.kappa in MHz; missing calibration requires input"
            )
        metadata[("tones", index, "drive")] = seed_metadata(
            f"qub_probe template bound to {name}; gain 0.1 and length 1.0 us candidates",
            fallback=True,
        )
    return Form(
        TwotoneFluxGlobalCfg,
        tree={
            "reset": Slot("$mux.module.reset"),
            "readout": mux_readout_initial(ctx),
            "tones": mux_tone_initials(tones, participants, fields=("drive", "freq")),
            "flux": Range(None, None, expts=101),
            "expts": 301,
            "participants": [index for index in participants],
            "settle": 0.1,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": 1.0,
            "relax_delay": 1.0,
        },
        initial_values=partial(global_flux_initial_values, names=names),
        fixed_elements=(
            FixedElements(("tones",), "name", ("participants",)),
            READOUT_ELEMENTS,
        ),
    ).with_seed_metadata(metadata)


def global_flux_initial_values(
    env: FormEnv, *, names: tuple[str, ...]
) -> dict[str, JsonValue]:
    """Seed the shared coil, native unit and union of captured Q intervals.

    env contains detached group parameters and DC infos; names retains the full
    captured Q order. Missing calibration leaves both bounds blank. Invalid
    device names, unavailable DC infos or invalid contributors raise without I/O.
    """
    group = require_mux_group(env.parameters)
    seeds = [captured_flux_seed(env, name, group.flux_dev) for name in names]
    start, stop = global_flux_interval(
        [None if seed.points is None else seed.points.integer for seed in seeds],
        [None if seed.points is None else seed.points.half for seed in seeds],
    )
    return {
        "flux_dev": group.flux_dev,
        "flux_unit": seeds[0].unit,
        "flux": {"kind": "linrange", "start": start, "stop": stop, "n": 101},
    }
