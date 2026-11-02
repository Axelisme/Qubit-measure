"""Captured mux onetone and local DC form defaults."""

from pydantic import JsonValue
from zcu_tools.cfg_model import FieldMeta, FormNode, Range, Slot
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
    mux_readout_seed_metadata,
    mux_tone_initials,
)
from zcu_lab.v2.support.measure.seed_metadata import seed_metadata

from .core import OnetoneFluxLocalCfg


def make_form(ctx: FormContext) -> Form[OnetoneFluxLocalCfg]:
    """Capture full tone identities, readout and zipped local DC candidates.

    ctx supplies the fixed Open/Reset group, aliases and calibration values.
    Missing DC calibration leaves selected bounds blank. All identities and
    devices remain editable only within their existing field policies; the tone
    picker edits an original position, not the participant selection. Invalid
    group bindings or inconsistent native device units raise without live I/O.
    """
    names = captured_mux_names(ctx)
    participants = captured_mux_participants(ctx, names)
    readout = mux_readout_initial(ctx, wait_for_ringup=True)
    tones: list[dict[str, FormNode]] = [
        {"name": name, "flux": {"start": None, "stop": None}} for name in names
    ]
    metadata = mux_readout_seed_metadata(readout)
    metadata[("readout",)] = FieldMeta(
        description="Captured readout slot with ringup timing, or group-wired blank mux candidate"
    )
    for field, description in (
        ("reps", "Fixed 1000 repetitions per round candidate"),
        ("rounds", "Fixed 10 software rounds candidate"),
        ("initial_delay", "Fixed 1.0 us initial delay candidate"),
        ("relax_delay", "Fixed 1.0 us recovery delay candidate"),
        ("settle", "Fixed 0.1 s DC settling candidate; confirm for each driver"),
        ("flux_expts", "Shared 101-point zipped local DC scan candidate"),
    ):
        metadata[(field,)] = seed_metadata(description, fallback=True)
    for endpoint in ("start", "stop", "n"):
        metadata[("df", endpoint)] = seed_metadata(
            "Nominal common offset candidate -1..1 MHz, 101 points", fallback=True
        )
    for index, name in enumerate(names):
        metadata[("tones", index, "flux_dev")] = FieldMeta(
            description=f"Device name captured from {name}.flux_dev"
        )
        for endpoint in ("start", "stop"):
            metadata[("tones", index, "flux", endpoint)] = seed_metadata(
                f"Actual half/integer from captured {name}.flux on its flux_dev, accounting for other device biases; native device unit"
            )

    def initial_values(env: FormEnv) -> dict[str, JsonValue]:
        defaults = form.model.defaults(env)
        tones = defaults["tones"]
        if not isinstance(tones, list):
            raise TypeError("local tones defaults must be a list")
        seeds = [
            captured_flux_seed(env, name, captured_qubit(env, name).flux_dev)
            for name in names
        ]
        units = {seed.unit for seed in seeds if seed.unit is not None}
        if len(units) > 1:
            raise ValueError("Mux flux devices must share one native unit")
        for index, (tone, seed) in enumerate(zip(tones, seeds, strict=True)):
            if not isinstance(tone, dict):
                raise TypeError("local tone defaults must be mappings")
            start, stop = flux_bounds(seed.points, 0.1)
            tone["flux_dev"] = seed.device
            tone["flux"] = (
                {"start": start, "stop": stop} if index in participants else None
            )
        return {"tones": tones, "flux_unit": next(iter(units)) if units else None}

    form = Form(
        OnetoneFluxLocalCfg,
        tree={
            "reset": Slot("$mux.module.reset"),
            "readout": readout,
            "tones": mux_tone_initials(tones, participants, fields=("flux",)),
            "flux_expts": 101,
            "df": Range(-1.0, 1.0, expts=101),
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
    return form
