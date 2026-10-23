"""Captured DC ranges and spectroscopy defaults for the four mux flux forms."""

from zcu_tools.cfg_model import Expr, FieldMeta, FormNode, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import require_mux_group
from zcu_lab.v2._support.measure.mux_forms import (
    captured_mux_names,
    mux_readout_initial,
    mux_readout_seed_metadata,
)
from zcu_lab.v2._support.mux_flux import GlobalFluxCfg, LocalFluxCfg
from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg
from zcu_lab.v2.mux.twotone_spectrum.core import MuxFreqCfg


def make_flux_form[
    C: GlobalFluxCfg[MuxOnetoneCfg]
    | LocalFluxCfg[MuxOnetoneCfg]
    | GlobalFluxCfg[MuxFreqCfg]
    | LocalFluxCfg[MuxFreqCfg]
](
    ctx: FormContext,
    cfg_type: type[C],
    *,
    local: bool,
    twotone: bool,
) -> Form[C]:
    """Build a fixed full identity list and editable native-unit DC scan.

    ctx is the Open/Reset capture. cfg_type identifies one of the four cores;
    local/twotone choose its line and inner spectrum policies. All participants
    start selected. Missing calibration leaves required ranges blank. Local
    ranges use per-qubit Expr seeds; global uses the union of captured coil
    int/half intervals only when every selected qubit has both values.
    Readout defaults reuse mux ringup policy for onetone and slot policy for
    twotone. Values remain editable, uncalibrated candidates, not safety limits.
    """
    names = captured_mux_names(ctx)
    group = require_mux_group(ctx.parameters)
    readout = mux_readout_initial(ctx, wait_for_ringup=not twotone)
    spectrum: dict[str, FormNode] = {
        "modules": {
            "reset": Slot("$mux.module.reset") if twotone else Hidden(),
            "readout": readout,
        },
        "reps": 1000,
        "rounds": 10,
        "relax_delay": 1.0,
        "initial_delay": Hidden(),
    }
    fixed = [
        FixedElements(("tones",), "name", ("participants",)),
        FixedElements(("spectrum", "modules", "readout", "module", "tones")),
    ]
    if twotone:
        spectrum.update(
            {
                "tones": [
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
                "expts": 301,
                "participants": list(names),
            }
        )
        fixed.append(
            FixedElements(("spectrum", "tones"), "name", ("spectrum", "participants"))
        )
    else:
        spectrum.update(
            {
                "sweep": {"df": Range.between(-1.0, 1.0, expts=101)},
                "active_tones": Hidden(None),
            }
        )
    tree: dict[str, FormNode] = {
        "spectrum": spectrum,
        "participants": list(names),
        "flux_unit": "A",
        "settle": 0.1,
    }
    metadata = {
        ("spectrum", *path): value
        for path, value in mux_readout_seed_metadata(readout).items()
    }
    if local:
        tree["tones"] = [
            {
                "name": name,
                "flux_dev": captured_flux_source(ctx, f"{name}.flux_source"),
                "sweep": {
                    "start": Expr(f"1.1 * {name}.flux_int - 0.1 * {name}.flux_half"),
                    "stop": Expr(f"1.1 * {name}.flux_half - 0.1 * {name}.flux_int"),
                },
            }
            for name in names
        ]
        tree["flux_expts"] = 101
        for index, name in enumerate(names):
            metadata[("tones", index, "flux_dev")] = FieldMeta(
                description=f"Device name captured from {name}.flux_source"
            )
            for endpoint in ("start", "stop"):
                metadata[("tones", index, "sweep", endpoint)] = FieldMeta(
                    description=f"From {name}.flux_int and {name}.flux_half; own DC line",
                    calibration="uncalibrated",
                )
    else:
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
        tree["flux"] = Range.between(start, stop, expts=101)
        for endpoint in ("start", "stop"):
            metadata[("flux", endpoint)] = FieldMeta(
                description="Union of captured participant $mux.coil_flux.<Q>.int/half intervals",
                calibration="uncalibrated",
            )
    return Form(
        cfg_type,
        tree,
        fixed_elements=tuple(fixed),
        field_order=(
            "participants",
            "tones",
            "flux_expts" if local else "flux",
            "spectrum",
            "flux_unit",
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
