"""Captured DC ranges and spectroscopy defaults for the four mux flux forms."""

from functools import partial

from pydantic import JsonValue
from zcu_tools.cfg_model import Expr, FieldMeta, FormNode, Hidden, Range, Slot, Template
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2.mux.onetone.core import MuxFrequencyOffsetCfg
from zcu_lab.v2.support.measure.flux_defaults import (
    captured_flux_seed,
    captured_qubit,
    flux_bounds,
)
from zcu_lab.v2.support.measure.mux_bindings import require_mux_group
from zcu_lab.v2.support.measure.mux_forms import (
    captured_mux_names,
    captured_mux_participants,
    mux_readout_initial,
    mux_readout_seed_metadata,
    mux_tone_initials,
)
from zcu_lab.v2.support.mux_flux import FluxFrequencyCfg, GlobalFluxCfg, LocalFluxCfg

from .seed_metadata import seed_metadata


def make_flux_form[
    C: GlobalFluxCfg[MuxFrequencyOffsetCfg]
    | LocalFluxCfg[MuxFrequencyOffsetCfg]
    | GlobalFluxCfg[FluxFrequencyCfg]
    | LocalFluxCfg[FluxFrequencyCfg]
](
    ctx: FormContext,
    cfg_type: type[C],
    *,
    local: bool,
    twotone: bool,
) -> Form[C]:
    """Build a flat form from its Open/Reset capture, with no live source reads.

    cfg_type selects the concrete core; local/twotone select its DC/frequency
    policies. Order is participants, DC identities, modules, frequency/DC
    scans, averaging, timing. Missing calibration leaves required bounds blank.
    Local bounds use each Q's actual device coordinates; global bounds are
    their union on the captured group source. Units come from device infos.
    Refs retain stored settings; generated defaults describe their source paths.
    Only fixed-constant fallbacks are marked uncalibrated. Refresh preserves
    inputs and seed capture.
    """
    names = captured_mux_names(ctx)
    participants = captured_mux_participants(ctx, names)
    readout = mux_readout_initial(ctx, wait_for_ringup=not twotone)
    tree: dict[str, FormNode] = {
        "participants": list[FormNode](participants),
        "reset": Slot("$mux.module.reset") if twotone else Hidden(),
        "readout": readout,
        "reps": 1000,
        "rounds": 10,
        "relax_delay": 1.0,
        "initial_delay": 1.0,
        "settle": 0.1,
    }
    fixed = [
        FixedElements(("readout", "module", "tones")),
    ]
    metadata = mux_readout_seed_metadata(readout)
    for field, description in (
        ("reps", "Fixed 1000 repetitions per round candidate"),
        ("rounds", "Fixed 10 software rounds candidate"),
        ("relax_delay", "Fixed 1.0 us recovery delay candidate"),
        ("initial_delay", "Fixed 1.0 us initial delay candidate"),
        ("settle", "Fixed 0.1 s DC settling candidate; confirm for each driver"),
    ):
        metadata[(field,)] = seed_metadata(description, fallback=True)
    metadata[("readout",)] = FieldMeta(
        description="Captured $mux.module.readout slot, or readout/mux:blank from group wiring; confirm calibration"
    )
    if twotone:
        tree["frequency"] = {
            "tones": mux_tone_initials(
                [
                    {
                        "name": name,
                        "drive": Template(
                            "qub_probe",
                            bindings={"qubit": name},
                            set={
                                "freq": Hidden(0.0),
                                "gain": 0.1,
                                "waveform.length": 1.0,
                            },
                        ),
                        "sweep": {
                            "start": Expr(f"{name}.freq - 1.5 * {name}.kappa"),
                            "stop": Expr(f"{name}.freq + 1.5 * {name}.kappa"),
                        },
                    }
                    for name in names
                ],
                participants,
                fields=("drive", "sweep"),
            ),
            "expts": 301,
        }
        fixed.append(
            FixedElements(
                ("frequency", "tones"),
                "name",
                ("participants",),
            )
        )
        metadata[("frequency", "expts")] = seed_metadata(
            "Shared 301-point hardware frequency scan candidate", fallback=True
        )
        for index, name in enumerate(names):
            for endpoint in ("start", "stop"):
                metadata[("frequency", "tones", index, "sweep", endpoint)] = (
                    seed_metadata(
                        f"From captured {name}.freq and {name}.kappa in MHz; missing calibration requires input"
                    )
                )
            metadata[("frequency", "tones", index, "drive")] = seed_metadata(
                f"qub_probe template bound to {name}; gain 0.1 and length 1.0 us candidates",
                fallback=True,
            )
    else:
        tree["frequency"] = {"df": Range(-1.0, 1.0, expts=101)}
        for endpoint in ("start", "stop", "n"):
            metadata[("frequency", "df", endpoint)] = seed_metadata(
                "Nominal common offset candidate -1..1 MHz, 101 points", fallback=True
            )
    if local:
        fixed.append(FixedElements(("tones",), "name", ("participants",)))
        tree["tones"] = mux_tone_initials(
            [
                {
                    "name": name,
                    "sweep": {"start": None, "stop": None},
                }
                for name in names
            ],
            participants,
            fields=("sweep",),
        )
        tree["flux_expts"] = 101
        metadata[("flux_expts",)] = seed_metadata(
            "Shared 101-point zipped local DC scan candidate", fallback=True
        )
        for index, name in enumerate(names):
            metadata[("tones", index, "flux_dev")] = FieldMeta(
                description=f"Device name captured from {name}.flux_dev"
            )
            for endpoint in ("start", "stop"):
                metadata[("tones", index, "sweep", endpoint)] = seed_metadata(
                    f"Actual half/integer from captured {name}.flux on its flux_dev, accounting for other device biases; native device unit"
                )
    else:
        fixed.append(FixedElements(("tones",), "name", ("participants",)))
        tree["tones"] = [{"name": name} for name in names]
        metadata[("flux_dev",)] = FieldMeta(
            description="Device name captured from $mux.flux_dev"
        )
        tree["flux"] = Range(None, None, expts=101)
        for endpoint in ("start", "stop"):
            metadata[("flux", endpoint)] = seed_metadata(
                "Union of actual half/integer intervals from each qubit.flux on $mux.flux_dev, accounting for captured other device biases; native device unit"
            )
        metadata[("flux", "n")] = seed_metadata(
            "Shared 101-point global DC scan candidate", fallback=True
        )
    return Form(
        cfg_type,
        tree,
        initial_values=partial(
            mux_flux_initial_values,
            names=tuple(names),
            participants=tuple(participants),
            local=local,
        ),
        fixed_elements=tuple(fixed),
        field_order=(
            "participants",
            "tones",
            "reset",
            "readout",
            "frequency",
            *(("flux_dev", "flux") if not local else ("flux_expts",)),
            "flux_unit",
            "reps",
            "rounds",
            "initial_delay",
            "relax_delay",
            "settle",
        ),
    ).with_seed_metadata(metadata)


def mux_flux_initial_values(
    env: FormEnv,
    *,
    names: tuple[str, ...],
    participants: tuple[int | str, ...],
    local: bool,
) -> dict[str, JsonValue]:
    """Seed mux DC identities, units and actual ranges from the fixed Form env.

    names is the full captured Q order; participants contains original selected
    indices. Local scans use each Q's flux_dev and leave nonparticipant
    sweeps disabled. Global scans use $mux.flux_dev and the union of Q
    intervals. Missing calibration leaves endpoints blank. Local devices must
    share one native unit. Invalid names, nonmember devices and frame/source
    failures propagate from captured_flux_seed without I/O or fallback.
    """
    group = require_mux_group(env.parameters)
    seeds = [
        captured_flux_seed(
            env,
            name,
            captured_qubit(env, name).flux_dev if local else group.flux_dev,
        )
        for name in names
    ]
    units = {seed.unit for seed in seeds if seed.unit is not None}
    if len(units) > 1:
        raise ValueError(
            f"Mux flux devices must share one native unit: {[(seed.device, seed.unit) for seed in seeds]!r}"
        )
    unit = next(iter(units)) if units else None
    if local:
        tones: list[JsonValue] = []
        for index, (name, seed) in enumerate(zip(names, seeds, strict=True)):
            start, stop = flux_bounds(seed.points, 0.1)
            tones.append(
                {
                    "name": name,
                    "flux_dev": seed.device,
                    "sweep": {"start": start, "stop": stop}
                    if index in participants
                    else None,
                }
            )
        return {"tones": tones, "flux_unit": unit}
    start, stop = global_flux_interval(
        [None if seed.points is None else seed.points.integer for seed in seeds],
        [None if seed.points is None else seed.points.half for seed in seeds],
    )
    return {
        "flux_dev": group.flux_dev,
        "flux_unit": unit,
        "flux": {"kind": "linrange", "start": start, "stop": stop, "n": 101},
    }


def global_flux_interval(
    integers: list[float | None] | None,
    halves: list[float | None] | None,
) -> tuple[float | None, float | None]:
    """Return the union of actual selected-device intervals, or two blanks.

    Both lists must align to the same selected qubits. Any absent calibration
    leaves both endpoints unfilled. Inputs are validated component fields.
    """
    if (
        integers is None
        or halves is None
        or any(value is None for value in (*integers, *halves))
    ):
        return None, None
    endpoints = [
        endpoint
        for integer, half in zip(integers, halves, strict=True)
        if integer is not None and half is not None
        for endpoint in (1.1 * integer - 0.1 * half, 1.1 * half - 0.1 * integer)
    ]
    return min(endpoints), max(endpoints)
