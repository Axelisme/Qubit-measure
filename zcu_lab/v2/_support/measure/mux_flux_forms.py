"""Captured DC ranges and spectroscopy defaults for the four mux flux forms."""

from zcu_tools.cfg_model import Expr, FieldMeta, FormNode, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import require_mux_group
from zcu_lab.v2._support.measure.mux_forms import (
    captured_mux_names,
    captured_mux_participants,
    mux_readout_initial,
    mux_readout_seed_metadata,
    mux_tone_initials,
)
from zcu_lab.v2._support.mux_flux import FluxFrequencyCfg, GlobalFluxCfg, LocalFluxCfg
from zcu_lab.v2.mux.onetone.core import MuxOnetoneSweepCfg

from .seed_metadata import seed_metadata


def make_flux_form[
    C: GlobalFluxCfg[MuxOnetoneSweepCfg]
    | LocalFluxCfg[MuxOnetoneSweepCfg]
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
    Local bounds are per-Q Expr seeds; global bounds are a captured union.
    Refs retain stored settings; generated defaults describe their source paths.
    Only fixed-constant fallbacks are marked uncalibrated. Refresh preserves
    inputs and seed capture.
    """
    names = captured_mux_names(ctx)
    participants = captured_mux_participants(ctx, names)
    group = require_mux_group(ctx.parameters)
    readout = mux_readout_initial(ctx, wait_for_ringup=not twotone)
    tree: dict[str, FormNode] = {
        "participants": list[FormNode](participants),
        "modules": {
            "reset": Slot("$mux.module.reset") if twotone else Hidden(),
            "readout": readout,
        },
        "reps": 1000,
        "rounds": 10,
        "relax_delay": 1.0,
        "initial_delay": 1.0,
        "flux_unit": "A",
        "settle": 0.1,
    }
    fixed = [
        FixedElements(("modules", "readout", "module", "tones")),
    ]
    metadata = mux_readout_seed_metadata(readout)
    for field, description in (
        ("reps", "Fixed 1000 repetitions per round candidate"),
        ("rounds", "Fixed 10 software rounds candidate"),
        ("relax_delay", "Fixed 1.0 us recovery delay candidate"),
        ("initial_delay", "Fixed 1.0 us initial delay candidate"),
        ("settle", "Fixed 0.1 s DC settling candidate; confirm for each driver"),
        ("flux_unit", "Initial A native-unit selection; must match connected drivers"),
    ):
        metadata[(field,)] = seed_metadata(description, fallback=True)
    metadata[("modules", "readout")] = FieldMeta(
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
                    "flux_dev": captured_flux_source(ctx, f"{name}.flux_source"),
                    "sweep": {
                        "start": Expr(
                            f"1.1 * {name}.flux_int - 0.1 * {name}.flux_half"
                        ),
                        "stop": Expr(f"1.1 * {name}.flux_half - 0.1 * {name}.flux_int"),
                    },
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
                description=f"Device name captured from {name}.flux_source"
            )
            for endpoint in ("start", "stop"):
                metadata[("tones", index, "sweep", endpoint)] = seed_metadata(
                    f"From {name}.flux_int and {name}.flux_half; own DC line in flux_unit, required when missing"
                )
    else:
        fixed.append(FixedElements(("tones",), "name", ("participants",)))
        tree["tones"] = [{"name": name} for name in names]
        tree["flux_dev"] = group.flux_source
        metadata[("flux_dev",)] = FieldMeta(
            description="Device name captured from $mux.flux_source"
        )
        calibrations = [group.coil_flux.get(name) for name in names]
        start, stop = global_flux_interval(
            [None if value is None else value.int for value in calibrations],
            [None if value is None else value.half for value in calibrations],
        )
        tree["flux"] = Range(start, stop, expts=101)
        for endpoint in ("start", "stop"):
            metadata[("flux", endpoint)] = seed_metadata(
                "Union of captured $mux.coil_flux.<qubit>.int/half intervals for $mux.qubits in flux_unit; missing calibration requires input"
            )
        metadata[("flux", "n")] = seed_metadata(
            "Shared 101-point global DC scan candidate", fallback=True
        )
    return Form(
        cfg_type,
        tree,
        fixed_elements=tuple(fixed),
        field_order=(
            "participants",
            "tones",
            "modules",
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


def captured_flux_source(ctx: FormContext, path: str) -> str | None:
    """Detach a captured device-name string; absent sources leave input blank.

    ctx is the Open/Reset source and path a concrete qubit flux_source field.
    Scalar Expr is numeric, so device names must be captured as literals.
    Invalid set names raise ValueError; source lookup errors propagate.
    """
    if ctx.parameters is None:
        raise ValueError("Flux forms require a parameter capture")
    source = ctx.parameters.lookup_point(path)
    if not source.present or source.value is None:
        return None
    if not isinstance(source.value, str) or not source.value.strip():
        raise ValueError(f"{path} must name a DC device")
    return source.value


def global_flux_interval(
    integers: list[float | None] | None,
    halves: list[float | None] | None,
) -> tuple[float | None, float | None]:
    """Return the union of captured selected coil intervals, or two blanks.

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
