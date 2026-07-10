"""Spec helper factories for common ReferenceSpec combinations."""

from __future__ import annotations

from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec
from zcu_tools.gui.app.main.specs.readout import (
    make_direct_readout_spec,
    make_pulse_readout_spec,
)
from zcu_tools.gui.app.main.specs.reset import (
    make_bath_reset_spec,
    make_none_reset_spec,
    make_pulse_reset_spec,
    make_two_pulse_reset_spec,
)
from zcu_tools.gui.cfg import (
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    FloatSpec,
    IntSpec,
    ReferenceSpec,
)


def make_readout_module_spec(
    label: str = "Readout", optional: bool = False
) -> ReferenceSpec:
    return ReferenceSpec(
        kind="module",
        allowed=[make_direct_readout_spec(), make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


def make_pulse_readout_module_spec(
    label: str = "Readout", optional: bool = False
) -> ReferenceSpec:
    return ReferenceSpec(
        kind="module",
        allowed=[make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


def make_pulse_module_spec(
    label: str = "Init Pulse", optional: bool = False
) -> ReferenceSpec:
    return ReferenceSpec(
        kind="module",
        allowed=[make_pulse_spec()],
        label=label,
        optional=optional,
    )


def make_reset_module_spec(
    label: str = "Reset", optional: bool = False
) -> ReferenceSpec:
    return ReferenceSpec(
        kind="module",
        allowed=[
            make_none_reset_spec(),
            make_pulse_reset_spec(),
            make_two_pulse_reset_spec(),
            make_bath_reset_spec(),
        ],
        label=label,
        optional=optional,
    )


# ---------------------------------------------------------------------------
# Single-shape reset module specs.
#
# These differ from ``make_reset_module_spec`` (the 4-shape ref used as an
# *optional* upstream reset in spectroscopy/Rabi adapters): a single-tone reset
# experiment fixes the *tested* reset to one concrete shape, so its ref allows
# exactly that shape. The form then shows no shape switcher for the tested
# reset — the experiment is intrinsically a one-shape calibration.
# ---------------------------------------------------------------------------


def make_pulse_reset_module_spec(
    label: str = "Tested Reset", optional: bool = False
) -> ReferenceSpec:
    """Single-shape ``reset/pulse`` ref (the tested reset of a one-pulse sweep)."""
    return ReferenceSpec(
        kind="module",
        allowed=[make_pulse_reset_spec()],
        label=label,
        optional=optional,
    )


def make_two_pulse_reset_module_spec(
    label: str = "Tested Reset", optional: bool = False
) -> ReferenceSpec:
    """Single-shape ``reset/two_pulse`` ref (the tested reset of a two-pulse sweep)."""
    return ReferenceSpec(
        kind="module",
        allowed=[make_two_pulse_reset_spec()],
        label=label,
        optional=optional,
    )


def make_bath_reset_module_spec(
    label: str = "Tested Reset", optional: bool = False
) -> ReferenceSpec:
    """Single-shape ``reset/bath`` ref (the tested reset of a bath-reset sweep)."""
    return ReferenceSpec(
        kind="module",
        allowed=[make_bath_reset_spec()],
        label=label,
        optional=optional,
    )


# ---------------------------------------------------------------------------
# Root cfg-spec assembly — the canonical top-level field order lives here, so
# adapters never hand-order it (the order-inconsistency bug class is owned by
# build_exp_spec, not repeated per adapter).
# ---------------------------------------------------------------------------

# Standard scalars shared by (almost) every experiment. Frozen → safe as
# defaults. An adapter overrides only when it differs (lookback locks reps=1).
DEFAULT_RELAX_DELAY_SPEC: CfgNodeSpec = FloatSpec(label="Relax delay (us)", decimals=3)
DEFAULT_REPS_SPEC: CfgNodeSpec = IntSpec(label="Reps")
DEFAULT_ROUNDS_SPEC: CfgNodeSpec = IntSpec(label="Rounds")


def declare_modules_spec(fields: dict[str, CfgNodeSpec]) -> CfgSectionSpec:
    """A "Modules" section wrapping the given module-ref fields."""
    return CfgSectionSpec(label="Modules", fields=fields)


def declare_sweep_spec(
    fields: dict[str, CfgNodeSpec], label: str = "Sweep"
) -> CfgSectionSpec:
    """A sweep section wrapping the given sweep-axis fields (default label "Sweep")."""
    return CfgSectionSpec(label=label, fields=fields)


def declare_dev_spec(
    fields: dict[str, CfgNodeSpec], label: str = "Device"
) -> CfgSectionSpec:
    """A device section wrapping the given device-ref fields."""
    return CfgSectionSpec(label=label, fields=fields)


def build_exp_spec(
    *,
    modules: dict[str, CfgNodeSpec],
    sweep: dict[str, CfgNodeSpec] | None = None,
    sweep_label: str = "Sweep",
    dev: dict[str, CfgNodeSpec] | None = None,
    extra: dict[str, CfgNodeSpec] | None = None,
    relax_delay: bool = True,
    reps: CfgNodeSpec = DEFAULT_REPS_SPEC,
    rounds: CfgNodeSpec = DEFAULT_ROUNDS_SPEC,
) -> CfgSectionSpec:
    """Assemble an experiment's root cfg spec in the canonical field order.

    Emits the top-level fields in one fixed order so adapters never hand-order
    them — the order is owned here, not per adapter:

        modules, [dev], [relax_delay], [sweep], [*extra], reps, rounds

    - ``modules`` (required): module-ref fields → the "Modules" section.
    - ``dev``: device-ref fields → the "Device" section (omit when None).
    - ``relax_delay``: add the standard relax-delay scalar; set ``False`` for an
      experiment that has no relax-delay field (e.g. the fake onetone sweep).
    - ``sweep`` / ``sweep_label``: sweep-axis fields → a section (omit the whole
      section when None, e.g. lookback). ``sweep_label`` overrides the "Sweep"
      header (e.g. an optimizer's "Search bounds (min–max)").
    - ``extra``: adapter-defined top-level knobs placed between sweep and reps.
      These can be real ExpCfg fields that lower normally or run-only knobs the
      adapter pops before lowering (e.g. ``earlystop_snr`` / ``num_points``).
    - ``reps`` / ``rounds``: default to the standard int scalars; pass a
      ``LiteralSpec`` to lock one (lookback locks reps to 1).
    """
    fields: dict[str, CfgNodeSpec] = {"modules": declare_modules_spec(modules)}
    if dev is not None:
        fields["dev"] = declare_dev_spec(dev)
    if relax_delay:
        fields["relax_delay"] = DEFAULT_RELAX_DELAY_SPEC
    if sweep is not None:
        fields["sweep"] = declare_sweep_spec(sweep, label=sweep_label)
    if extra:
        fields.update(extra)
    fields["reps"] = reps
    fields["rounds"] = rounds
    return CfgSectionSpec(fields=fields)


def schema_from_module(proposed: object | None) -> CfgSchema | None:
    """Edit CfgSchema from a proposed module/waveform cfg (module-agnostic).

    Delegates to ``module_cfg_to_value``, which auto-routes waveform cfgs to
    ``waveform_cfg_to_value``. Returns None when ``proposed`` is None (no edit
    button shown). Unknown module types fast-fail (``module_cfg_to_value``
    raises) — deliberate.
    """
    if proposed is None:
        return None
    spec, value = module_cfg_to_value(proposed)
    return CfgSchema(spec=spec, value=value)
