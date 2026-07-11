"""Closed catalog for program module and waveform GUI shapes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from zcu_tools.gui.cfg import CfgSectionSpec

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


# Import the constructors only after defining ProgramSpecPolicy.  The spec module
# imports that policy type but never the catalog instance, keeping construction
# acyclic and the vocabulary list in this file.
from .specs import (  # noqa: E402
    _make_arb_waveform_spec,
    _make_bath_reset_spec,
    _make_const_waveform_spec,
    _make_cosine_waveform_spec,
    _make_direct_readout_spec,
    _make_drag_waveform_spec,
    _make_flat_top_waveform_spec,
    _make_gauss_waveform_spec,
    _make_none_reset_spec,
    _make_pulse_readout_spec,
    _make_pulse_reset_spec,
    _make_pulse_spec,
    _make_two_pulse_reset_spec,
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
        ProgramShape("waveform", "arb", "Arb", _make_arb_waveform_spec),
        ProgramShape("waveform", "flat_top", "FlatTop", _make_flat_top_waveform_spec),
    )
)
