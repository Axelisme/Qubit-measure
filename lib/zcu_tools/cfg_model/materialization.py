"""Pure Spec validation and dependency-injected materialization entry points."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from .base import ConfigBase
from .dependencies import CfgValidationContext
from .inputs import CfgInputMap
from .module import (
    BathResetSpec,
    DirectReadoutCfg,
    ModuleCfg,
    ModuleSpec,
    NoneResetCfg,
    PulseCfg,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    TwoPulseResetSpec,
)
from .waveform import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformSpec,
    GaussWaveformCfg,
    RaiseWaveformCfg,
    WaveformCfg,
    WaveformSpec,
)


def _isolated_raw(value: object) -> object:
    """Copy input fields, stripping prior validation and private dependency bindings."""
    if isinstance(value, ConfigBase):
        return _isolated_raw(dict(value))
    if isinstance(value, dict):
        return {key: _isolated_raw(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_isolated_raw(child) for child in value]
    return deepcopy(value)


def _persistent_raw(value: object) -> object:
    """Reject runtime state before copying or recursively dumping Spec fields."""
    if isinstance(value, ConfigBase):
        if not isinstance(
            value,
            (
                PulseSpec,
                PulseReadoutSpec,
                PulseResetSpec,
                TwoPulseResetSpec,
                BathResetSpec,
                DirectReadoutCfg,
                NoneResetCfg,
                FlatTopWaveformSpec,
                ConstWaveformCfg,
                CosineWaveformCfg,
                GaussWaveformCfg,
                DragWaveformCfg,
                ArbWaveformCfg,
            ),
        ):
            raise ValueError(f"Not a persistent Spec: {type(value).__name__}")
        if isinstance(value, ArbWaveformCfg) and value.has_asset_binding:
            raise ValueError("Not a persistent Spec: bound ArbWaveformCfg")
        # Pydantic iteration keeps child objects, unlike recursive model_dump.
        return _persistent_raw(dict(value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Not a persistent value: mapping keys must be strings")
        return {key: _persistent_raw(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_persistent_raw(child) for child in value]
    if type(value) in (type(None), bool, int, float, str):
        return value
    raise ValueError(f"Not a persistent value: {type(value).__name__}")


def _resolution_error(
    root: str, chain: tuple[str, ...], expected: str, reason: str
) -> ValueError:
    path = " -> ".join(chain) or "<inline>"
    return ValueError(f"Root {root}; chain {path}; expected {expected}: {reason}")


def _expand_waveform(
    raw: object,
    context: CfgValidationContext,
    root: str,
    chain: tuple[str, ...] = (),
    *,
    expected: Literal["waveform", "ramp"] = "waveform",
) -> WaveformCfg:
    if isinstance(raw, str):
        key = raw
        chain = (*chain, key)
        if key in chain[:-1]:
            raise _resolution_error(root, chain, expected, "Cyclic reference")
        if context.modules is None:
            raise _resolution_error(root, chain, expected, "Missing modules dependency")
        try:
            raw = context.modules.waveform_spec(key)
        except (ValueError, KeyError) as exc:
            raise _resolution_error(root, chain, expected, str(exc)) from exc

    try:
        data = TypeAdapter(dict[str, object]).validate_python(_isolated_raw(raw))
        if data.get("style") == "flat_top" and "raise_waveform" in data:
            data["raise_waveform"] = _expand_waveform(
                data["raise_waveform"], context, root, chain, expected="ramp"
            )
        if expected == "ramp":
            return TypeAdapter(RaiseWaveformCfg).validate_python(data, context=context)
        return TypeAdapter(WaveformCfg).validate_python(data, context=context)
    except ValidationError as exc:
        raise _resolution_error(root, chain, expected, str(exc)) from exc


def _expand_module(
    raw: object,
    context: CfgValidationContext,
    root: str,
    chain: tuple[str, ...] = (),
    *,
    expected: Literal["module", "pulse"] = "module",
) -> ModuleCfg:
    if isinstance(raw, str):
        key = raw
        chain = (*chain, key)
        if key in chain[:-1]:
            raise _resolution_error(root, chain, expected, "Cyclic reference")
        if context.modules is None:
            raise _resolution_error(root, chain, expected, "Missing modules dependency")
        try:
            raw = context.modules.module_spec(key)
        except (ValueError, KeyError) as exc:
            raise _resolution_error(root, chain, expected, str(exc)) from exc

    try:
        data = TypeAdapter(dict[str, object]).validate_python(_isolated_raw(raw))
        kind = data.get("type")
        if kind == "pulse" and "waveform" in data:
            data["waveform"] = _isolated_raw(
                _expand_waveform(data["waveform"], context, root, chain)
            )
        elif kind in ("readout/pulse", "reset/pulse") and "pulse_cfg" in data:
            data["pulse_cfg"] = _isolated_raw(
                _expand_module(
                    data["pulse_cfg"], context, root, chain, expected="pulse"
                )
            )
        elif kind in ("reset/two_pulse", "reset/bath"):
            fields = (
                ("pulse1_cfg", "pulse2_cfg")
                if kind == "reset/two_pulse"
                else ("cavity_tone_cfg", "qubit_tone_cfg", "pi2_cfg")
            )
            for field in fields:
                if field in data:
                    data[field] = _isolated_raw(
                        _expand_module(
                            data[field], context, root, chain, expected="pulse"
                        )
                    )
        if expected == "pulse":
            return TypeAdapter(PulseCfg).validate_python(data, context=context)
        return TypeAdapter(ModuleCfg).validate_python(data, context=context)
    except ValidationError as exc:
        raise _resolution_error(root, chain, expected, str(exc)) from exc


def module_spec_from_raw(raw: CfgInputMap | ModuleSpec) -> ModuleSpec:
    """Validate an isolated persistent module candidate without lookups or I/O.

    String children are concrete library keys and remain strings. Invalid fields,
    families, or non-YAML runtime values raise ValueError; no runtime cfg lowering
    or lossy serialization is performed.
    """
    return TypeAdapter(ModuleSpec).validate_python(_persistent_raw(raw))


def waveform_spec_from_raw(raw: CfgInputMap | WaveformSpec) -> WaveformSpec:
    """Validate an isolated persistent waveform candidate without lookups or I/O.

    String ramp keys remain strings. Invalid fields, ramp families, or non-YAML
    runtime values raise ValueError. Arb Specs need no asset provider.
    """
    return TypeAdapter(WaveformSpec).validate_python(_persistent_raw(raw))


class ModuleCfgFactory:
    """Materialize isolated module cfgs with a single reference-chain lifetime."""

    @staticmethod
    def from_raw(
        raw: str | CfgInputMap | ModuleSpec | ModuleCfg,
        *,
        context: CfgValidationContext | None = None,
    ) -> ModuleCfg:
        """Expand raw or a concrete key to a new validated module Cfg.

        context supplies modules for string references and assets for arb.
        Missing dependencies/keys, wrong families, cycles, and invalid fields
        raise ValueError; resolution failures include root, chain and family.
        Inputs and resolver snapshots are never mutated or trusted as validated.
        """
        root = raw if isinstance(raw, str) else "<inline>"
        return _expand_module(raw, context or CfgValidationContext(), root)


class WaveformCfgFactory:
    """Materialize isolated waveform cfgs with explicit library and asset owners."""

    @staticmethod
    def from_raw(
        raw: str | CfgInputMap | WaveformSpec | WaveformCfg,
        *,
        context: CfgValidationContext | None = None,
    ) -> WaveformCfg:
        """Expand raw or a concrete key to a new validated waveform Cfg.

        context supplies modules for keys and assets for arb. Inline scalar cfgs
        need no context. Invalid fields/families, cycles or missing dependencies
        raise ValueError with root, chain and expected-family resolution evidence.
        Inputs and resolver snapshots are not mutated; runtime children revalidate.
        """
        root = raw if isinstance(raw, str) else "<inline>"
        return _expand_waveform(raw, context or CfgValidationContext(), root)
