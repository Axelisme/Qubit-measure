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
    CfgSchema,
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
