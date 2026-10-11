"""Closed catalog for program module and waveform GUI shapes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
    make_default_value,
)

ProgramCfgKind = Literal["module", "waveform"]


@dataclass(frozen=True, slots=True)
class ProgramSpecPolicy:
    """The two intentional cross-app differences in program cfg specs."""

    arb_data_choices_source: str = ""
    enable_readout_shape_inheritance: bool = False


_SpecFactory = Callable[[ProgramSpecPolicy, str], CfgSectionSpec]


@dataclass(frozen=True, slots=True)
class ProgramShape:
    kind: ProgramCfgKind
    discriminator: str
    label: str
    _spec_factory: _SpecFactory = field(repr=False, compare=False)

    def make_spec(
        self, policy: ProgramSpecPolicy, *, label: str | None = None
    ) -> CfgSectionSpec:
        """Build a deep-fresh spec, optionally overriding its root label."""
        return self._spec_factory(policy, self.label if label is None else label)


class UnknownProgramShapeError(LookupError):
    """An explicit discriminator is outside the closed program vocabulary."""


@dataclass(frozen=True, slots=True, init=False)
class ProgramShapeCatalog:
    _modules: tuple[ProgramShape, ...]
    _waveforms: tuple[ProgramShape, ...]
    _by_kind: Mapping[ProgramCfgKind, Mapping[str, ProgramShape]]

    def __init__(self, shapes: tuple[ProgramShape, ...]) -> None:
        modules = tuple(shape for shape in shapes if shape.kind == "module")
        waveforms = tuple(shape for shape in shapes if shape.kind == "waveform")
        by_kind: Mapping[ProgramCfgKind, Mapping[str, ProgramShape]] = MappingProxyType(
            {
                "module": MappingProxyType(
                    {shape.discriminator: shape for shape in modules}
                ),
                "waveform": MappingProxyType(
                    {shape.discriminator: shape for shape in waveforms}
                ),
            }
        )
        object.__setattr__(self, "_modules", modules)
        object.__setattr__(self, "_waveforms", waveforms)
        object.__setattr__(self, "_by_kind", by_kind)

    def module(self, discriminator: str) -> ProgramShape:
        return self.get("module", discriminator)

    def waveform(self, style: str) -> ProgramShape:
        return self.get("waveform", style)

    def get(self, kind: ProgramCfgKind, discriminator: str) -> ProgramShape:
        shapes = self._by_kind[kind]
        shape = shapes.get(discriminator)
        if shape is None:
            allowed = ", ".join(shapes)
            raise UnknownProgramShapeError(
                f"Unknown {kind} program shape {discriminator!r}; allowed: {allowed}"
            )
        return shape

    def modules(self) -> tuple[ProgramShape, ...]:
        return self._modules

    def waveforms(self) -> tuple[ProgramShape, ...]:
        return self._waveforms


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
                discriminator="style",
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
                discriminator="style",
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
                discriminator="type",
                allowed=[_make_pulse_spec(policy, "Pulse")],
                label="Cavity Tone",
            ),
            "qubit_tone_cfg": ReferenceSpec(
                kind="module",
                discriminator="type",
                allowed=[_make_pulse_spec(policy, "Pulse")],
                label="Qubit Tone",
            ),
            "pi2_cfg": ReferenceSpec(
                kind="module",
                discriminator="type",
                allowed=[_make_pulse_spec(policy, "Pulse")],
                label="Pi/2 Pulse",
            ),
        },
    )


PROGRAM_SHAPES = ProgramShapeCatalog(
    (
        ProgramShape("module", "pulse", "Pulse", _make_pulse_spec),
        ProgramShape(
            "module",
            "readout/direct",
            "Direct Readout",
            _make_direct_readout_spec,
        ),
        ProgramShape(
            "module", "readout/pulse", "Pulse Readout", _make_pulse_readout_spec
        ),
        ProgramShape("module", "reset/none", "None Reset", _make_none_reset_spec),
        ProgramShape("module", "reset/pulse", "Pulse Reset", _make_pulse_reset_spec),
        ProgramShape(
            "module",
            "reset/two_pulse",
            "Two-Pulse Reset",
            _make_two_pulse_reset_spec,
        ),
        ProgramShape("module", "reset/bath", "Bath Reset", _make_bath_reset_spec),
        ProgramShape("waveform", "const", "Const", _make_const_waveform_spec),
        ProgramShape("waveform", "cosine", "Cosine", _make_cosine_waveform_spec),
        ProgramShape("waveform", "gauss", "Gauss", _make_gauss_waveform_spec),
        ProgramShape("waveform", "drag", "DRAG", _make_drag_waveform_spec),
        ProgramShape("waveform", "flat_top", "FlatTop", _make_flat_top_waveform_spec),
        ProgramShape("waveform", "arb", "Arb", _make_arb_waveform_spec),
    )
)


_MISSING_DISCRIMINATOR = object()


def program_shape_for_input(
    kind: ProgramCfgKind,
    cfg_input: object,
) -> ProgramShape:
    """Inspect one root discriminator without normalizing or materializing cfg."""

    if kind == "module":
        key = "type"
    elif kind == "waveform":
        key = "style"
    else:
        raise RuntimeError(f"Unsupported program reference kind {kind!r}")

    if isinstance(cfg_input, Mapping):
        discriminator = cfg_input.get(key, _MISSING_DISCRIMINATOR)
    else:
        discriminator = getattr(cfg_input, key, _MISSING_DISCRIMINATOR)
    if discriminator is _MISSING_DISCRIMINATOR:
        raise ValueError(f"Program {kind} input is missing discriminator {key!r}")
    if not isinstance(discriminator, str):
        raise TypeError(
            f"Program discriminator {key!r} must be str, "
            f"got {type(discriminator).__name__}"
        )
    return PROGRAM_SHAPES.get(kind, discriminator)
