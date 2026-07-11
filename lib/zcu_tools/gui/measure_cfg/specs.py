"""Fresh program module and waveform GUI spec construction.

This module owns only the measure-domain projection of the program/v2 cfg
vocabulary.  It deliberately depends on the generic cfg model, not on either GUI
application or the program runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
    make_default_value,
)

if TYPE_CHECKING:
    from .catalog import ProgramSpecPolicy


def _make_const_waveform_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    del policy
    return CfgSectionSpec(
        label=label,
        fields={
            "style": LiteralSpec("const"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        },
    )


def _make_cosine_waveform_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    del policy
    return CfgSectionSpec(
        label=label,
        fields={
            "style": LiteralSpec("cosine"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        },
    )


def _make_gauss_waveform_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    del policy
    return CfgSectionSpec(
        label=label,
        fields={
            "style": LiteralSpec("gauss"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "sigma": ScalarSpec(label="Sigma (us)", type=float, decimals=3),
        },
    )


def _make_drag_waveform_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    del policy
    return CfgSectionSpec(
        label=label,
        fields={
            "style": LiteralSpec("drag"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "sigma": ScalarSpec(label="Sigma (us)", type=float, decimals=3),
            "delta": ScalarSpec(label="Delta (MHz)", type=float, decimals=2),
            "alpha": ScalarSpec(label="Alpha", type=float, decimals=4),
        },
    )


def _make_arb_waveform_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    return CfgSectionSpec(
        label=label,
        fields={
            "style": LiteralSpec("arb"),
            "data": ScalarSpec(
                label="Data key",
                type=str,
                choices_source=policy.arb_data_choices_source,
            ),
        },
    )


def _make_flat_top_waveform_spec(
    policy: ProgramSpecPolicy, label: str
) -> CfgSectionSpec:
    return CfgSectionSpec(
        label=label,
        fields={
            "style": LiteralSpec("flat_top"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "raise_waveform": ReferenceSpec(
                kind="waveform",
                allowed=[
                    _make_cosine_waveform_spec(policy, "Cosine"),
                    _make_gauss_waveform_spec(policy, "Gauss"),
                    _make_drag_waveform_spec(policy, "DRAG"),
                    _make_arb_waveform_spec(policy, "Arb"),
                ],
                label="Raise Waveform",
            ),
        },
    )


def _make_pulse_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    return CfgSectionSpec(
        label=label,
        fields={
            "type": LiteralSpec("pulse"),
            "waveform": ReferenceSpec(
                kind="waveform",
                allowed=[
                    _make_const_waveform_spec(policy, "Const"),
                    _make_cosine_waveform_spec(policy, "Cosine"),
                    _make_gauss_waveform_spec(policy, "Gauss"),
                    _make_drag_waveform_spec(policy, "DRAG"),
                    _make_arb_waveform_spec(policy, "Arb"),
                    _make_flat_top_waveform_spec(policy, "FlatTop"),
                ],
                label="Waveform",
            ),
            "ch": ScalarSpec(label="Gen ch", type=int),
            "nqz": ScalarSpec(label="NQZ", type=int, choices=[1, 2], group="Advanced"),
            "freq": ScalarSpec(label="Freq (MHz)", type=float, decimals=2),
            "gain": ScalarSpec(label="Gain", type=float, decimals=4),
            "phase": ScalarSpec(
                label="Phase (deg)", type=float, decimals=2, group="Advanced"
            ),
            "pre_delay": ScalarSpec(
                label="Pre-delay (us)", type=float, decimals=3, group="Advanced"
            ),
            "post_delay": ScalarSpec(
                label="Post-delay (us)", type=float, decimals=3, group="Advanced"
            ),
            "mixer_freq": ScalarSpec(
                label="Mixer freq (MHz)",
                type=float,
                decimals=2,
                optional=True,
                group="Advanced",
            ),
        },
    )


def _inherit_direct_readout(
    old_val: CfgSectionValue, old_spec: CfgSectionSpec
) -> CfgSectionValue | None:
    if old_spec.label != "Pulse Readout":
        return None
    ro_cfg_val = old_val.fields.get("ro_cfg")
    if isinstance(ro_cfg_val, CfgSectionValue):
        return ro_cfg_val
    return None


def _inherit_pulse_readout(
    old_val: CfgSectionValue,
    old_spec: CfgSectionSpec,
    *,
    policy: ProgramSpecPolicy,
) -> CfgSectionValue | None:
    if old_spec.label != "Direct Readout":
        return None
    result = make_default_value(_make_pulse_readout_spec(policy, "Pulse Readout"))
    result.fields["ro_cfg"] = old_val
    return result


@dataclass(frozen=True, slots=True)
class _PulseReadoutInheritance:
    """Deepcopy-stable callable carrying the spec policy used for the new shape."""

    policy: ProgramSpecPolicy

    def __call__(
        self, old_val: CfgSectionValue, old_spec: CfgSectionSpec
    ) -> CfgSectionValue | None:
        return _inherit_pulse_readout(old_val, old_spec, policy=self.policy)


def _make_direct_readout_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    inherit_hook = (
        _inherit_direct_readout if policy.enable_readout_shape_inheritance else None
    )
    return CfgSectionSpec(
        label=label,
        inherit_hook=inherit_hook,
        fields={
            "type": LiteralSpec("readout/direct"),
            "ro_ch": ScalarSpec(label="RO ch", type=int),
            "ro_freq": ScalarSpec(label="RO Freq (MHz)", type=float, decimals=2),
            "ro_length": ScalarSpec(label="RO length (us)", type=float, decimals=3),
            "trig_offset": ScalarSpec(label="Trig offset (us)", type=float, decimals=3),
            "gen_ch": ScalarSpec(
                label="Gen ch", type=int, optional=True, group="Advanced"
            ),
        },
    )


def _make_pulse_readout_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    inherit_hook = (
        _PulseReadoutInheritance(policy)
        if policy.enable_readout_shape_inheritance
        else None
    )
    return CfgSectionSpec(
        label=label,
        inherit_hook=inherit_hook,
        fields={
            "type": LiteralSpec("readout/pulse"),
            "pulse_cfg": _make_pulse_spec(policy, "Pulse"),
            "ro_cfg": _make_direct_readout_spec(policy, "Direct Readout"),
        },
    )


def _make_none_reset_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    del policy
    return CfgSectionSpec(
        label=label,
        fields={"type": LiteralSpec("reset/none")},
    )


def _make_pulse_reset_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    return CfgSectionSpec(
        label=label,
        fields={
            "type": LiteralSpec("reset/pulse"),
            "pulse_cfg": _make_pulse_spec(policy, "Pulse"),
        },
    )


def _make_two_pulse_reset_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    return CfgSectionSpec(
        label=label,
        fields={
            "type": LiteralSpec("reset/two_pulse"),
            "pulse1_cfg": _make_pulse_spec(policy, "Pulse 1"),
            "pulse2_cfg": _make_pulse_spec(policy, "Pulse 2"),
        },
    )


def _make_bath_reset_spec(policy: ProgramSpecPolicy, label: str) -> CfgSectionSpec:
    return CfgSectionSpec(
        label=label,
        fields={
            "type": LiteralSpec("reset/bath"),
            "cavity_tone_cfg": ReferenceSpec(
                kind="module",
                allowed=[_make_pulse_spec(policy, "Pulse")],
                label="Cavity Tone",
            ),
            "qubit_tone_cfg": ReferenceSpec(
                kind="module",
                allowed=[_make_pulse_spec(policy, "Pulse")],
                label="Qubit Tone",
            ),
            "pi2_cfg": ReferenceSpec(
                kind="module",
                allowed=[_make_pulse_spec(policy, "Pulse")],
                label="Pi/2 Pulse",
            ),
        },
    )
