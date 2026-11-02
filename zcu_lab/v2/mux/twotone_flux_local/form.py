"""Captured mux twotone drives, frequency ranges and zipped local DC inputs."""

from typing import Any

from zcu_tools.cfg_model import Expr, FieldMeta, FormNode, Hidden, Slot, Template
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2.support.measure.flux_defaults import (
    captured_flux_seed,
    captured_qubit,
    flux_bounds,
)
from zcu_lab.v2.support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_names,
    captured_mux_participants,
    mux_readout_initial,
    mux_tone_initials,
)
from zcu_lab.v2.support.measure.seed_metadata import seed_metadata

from .core import TwotoneFluxLocalCfg


def make_form(ctx: FormContext) -> Form[TwotoneFluxLocalCfg]:
    """Build a single full tone list with drives, MHz ranges and local DC ranges.

    ctx supplies the Open/Reset mux identities and calibration capture. Selected
    tones start with probe templates and frequency expressions; other tones'
    nullable drive/range fields start disabled but remain editable. The factory
    adds each captured local device, its projected DC bounds and common unit.
    Missing calibration leaves required bounds blank. Invalid captured sources
    and inconsistent native units raise; no live device reads occur.
    """
    names = captured_mux_names(ctx)
    participants = captured_mux_participants(ctx, names)
    tones: list[dict[str, FormNode]] = [
        {
            "name": name,
            "drive": Template(
                "qub_probe",
                bindings={"qubit": name},
                set={"freq": Hidden(0.0), "gain": 0.1, "waveform.length": 1.0},
            ),
            "freq": {
                "start": Expr(f"{name}.freq - 1.5 * {name}.kappa"),
                "stop": Expr(f"{name}.freq + 1.5 * {name}.kappa"),
            },
            "flux": {"start": None, "stop": None},
        }
        for name in names
    ]

    def initial_values(env: FormEnv, defaults: dict[str, Any]) -> dict[str, Any]:
        """Add captured DC values to fresh declared raw input, preserving drive seeds."""
        seeds = [
            captured_flux_seed(env, name, captured_qubit(env, name).flux_dev)
            for name in names
        ]
        units = {seed.unit for seed in seeds if seed.unit is not None}
        if len(units) > 1:
            raise ValueError(
                f"Mux flux devices must share one native unit: {[(seed.device, seed.unit) for seed in seeds]!r}"
            )
        for index, (tone, seed) in enumerate(
            zip(defaults["tones"], seeds, strict=True)
        ):
            start, stop = flux_bounds(seed.points, 0.1)
            tone["flux_dev"] = seed.device
            tone["flux"] = (
                {"start": start, "stop": stop} if index in participants else None
            )
        defaults["flux_unit"] = next(iter(units)) if units else None
        return defaults

    metadata = {
        ("readout",): FieldMeta(
            description="Captured $mux.module.readout slot, or readout/mux:blank from group wiring; confirm calibration"
        ),
    }
    for field, description in (
        ("reps", "Fixed 1000 repetitions per round candidate"),
        ("rounds", "Fixed 10 software rounds candidate"),
        ("relax_delay", "Fixed 1.0 us recovery delay candidate"),
        ("initial_delay", "Fixed 1.0 us initial delay candidate"),
        ("settle", "Fixed 0.1 s DC settling candidate; confirm for each driver"),
        ("expts", "Shared 301-point hardware frequency scan candidate"),
        ("flux_expts", "Shared 101-point zipped local DC scan candidate"),
    ):
        metadata[(field,)] = seed_metadata(description, fallback=True)
    for index, name in enumerate(names):
        metadata[("tones", index, "drive")] = seed_metadata(
            f"qub_probe template bound to {name}; gain 0.1 and length 1.0 us candidates",
            fallback=True,
        )
        metadata[("tones", index, "flux_dev")] = FieldMeta(
            description=f"Device name captured from {name}.flux_dev"
        )
        for endpoint in ("start", "stop"):
            metadata[("tones", index, "freq", endpoint)] = seed_metadata(
                f"From captured {name}.freq and {name}.kappa in MHz; missing calibration requires input"
            )
            metadata[("tones", index, "flux", endpoint)] = seed_metadata(
                f"Actual half/integer from captured {name}.flux on its flux_dev, accounting for other device biases; native device unit"
            )
    return Form(
        TwotoneFluxLocalCfg,
        tree={
            "reset": Slot("$mux.module.reset"),
            "readout": mux_readout_initial(ctx),
            "tones": mux_tone_initials(
                tones, participants, fields=("drive", "freq", "flux")
            ),
            "flux_expts": 101,
            "expts": 301,
            "participants": list[FormNode](participants),
            "settle": 0.1,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": 1.0,
            "relax_delay": 1.0,
        },
        initial_values=initial_values,
        fixed_elements=(
            FixedElements(("tones",), "name", ("participants",)),
            READOUT_ELEMENTS,
        ),
    ).with_seed_metadata(metadata)
