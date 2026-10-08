"""Waveform schemas without runtime builders or repository lookups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, PrivateAttr
from qick.asm_v2 import QickParam

from .base import ConfigBase
from .dependencies import CfgValidationContext, WaveformAssetInfo, WaveformSamples


class AbsWaveformCfg(ConfigBase, ABC):
    """Waveform parameters; style identifies the concrete envelope family."""

    style: str

    @abstractmethod
    def set_param(self, name: str, value: float | QickParam) -> None:
        """Update an envelope knob; invalid names or unsupported sweeps raise ValueError."""


class ConstWaveformCfg(AbsWaveformCfg):
    """Constant envelope. length is its duration in µs, optionally a QICK sweep."""

    style: Literal["const"] = "const"
    length: float | QickParam

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set length in µs, accepting a sweep; any other name raises ValueError."""
        if name == "length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class CosineWaveformCfg(AbsWaveformCfg):
    """Cosine envelope. length is a fixed duration in µs; sweeps are unsupported."""

    style: Literal["cosine"] = "cosine"
    length: float

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set fixed length in µs; unknown names and QickParam raise ValueError."""
        if name != "length":
            raise ValueError(f"Unknown parameter: {name}")
        if isinstance(value, QickParam):
            raise ValueError("Cosine waveform length must be a float value")
        self.length = value


class GaussWaveformCfg(AbsWaveformCfg):
    """Gaussian envelope. length and sigma are fixed duration and width in µs."""

    style: Literal["gauss"] = "gauss"
    length: float
    sigma: float

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set length (rescaling sigma), only_length, or sigma in µs.

        Unknown names and QickParam raise ValueError.
        """
        if isinstance(value, QickParam):
            raise ValueError(f"Gauss waveform {name} must not be a QickParam")
        if name == "length":
            sigma_ratio = self.sigma / self.length
            self.length = value
            self.sigma = sigma_ratio * value
        elif name == "sigma":
            self.sigma = value
        elif name == "only_length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class DragWaveformCfg(AbsWaveformCfg):
    """DRAG Gaussian. length/sigma are in µs, delta in MHz, alpha is dimensionless.

    All four fields are fixed floats; QICK sweeps are unsupported.
    """

    style: Literal["drag"] = "drag"
    length: float
    sigma: float
    delta: float
    alpha: float

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set length (rescaling sigma), only_length, sigma, delta, or alpha.

        Units follow the fields. Unknown names and QickParam raise ValueError.
        """
        if isinstance(value, QickParam):
            raise ValueError(f"Drag waveform {name} must not be a QickParam")
        if name == "length":
            sigma_ratio = self.sigma / self.length
            self.length = value
            self.sigma = sigma_ratio * value
        elif name == "sigma":
            self.sigma = value
        elif name == "delta":
            self.delta = value
        elif name == "alpha":
            self.alpha = value
        elif name == "only_length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


@dataclass(frozen=True)
class _AssetBinding:
    """Keep one asset owner shared when a cfg is copied, without freezing its bytes."""

    provider: WaveformAssetInfo

    def __deepcopy__(self, _memo: dict[int, object]) -> _AssetBinding:
        # Copy cfg fields, not the injected owner or its live asset source.
        return self


class ArbWaveformCfg(AbsWaveformCfg):
    """Asset-backed envelope; data is a concrete asset key, not a file path.

    A document can exist without assets. Materialization binds an asset provider
    without serializing it. Reads use current bytes in that provider's scope.
    """

    style: Literal["arb"] = "arb"
    data: str
    _asset_binding: _AssetBinding | None = PrivateAttr(default=None)

    def model_post_init(self, context: object) -> None:
        """Bind typed materialization assets; document construction leaves assets unset.

        A materialization context without assets raises ValueError.
        """
        if isinstance(context, CfgValidationContext):
            if context.assets is None:
                raise ValueError("Arb waveform requires assets in CfgValidationContext")
            self._asset_binding = _AssetBinding(context.assets)

    @property
    def has_asset_binding(self) -> bool:
        """Whether materialization attached an asset owner; no I/O or serialization occurs."""
        return self._asset_binding is not None

    @property
    def length(self) -> float:
        """Return current reference duration in µs; missing assets raise ValueError."""
        if self._asset_binding is None:
            raise ValueError("Arb waveform requires assets in CfgValidationContext")
        return self._asset_binding.provider.duration(self.data)

    def asset_samples(self) -> WaveformSamples:
        """Read isolated current I/Q samples; propagate asset-owner read errors.

        Missing materialization binding raises ValueError.
        """
        if self._asset_binding is None:
            raise ValueError("Arb waveform requires assets in CfgValidationContext")
        return self._asset_binding.provider.samples(self.data)

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Reject every knob with ValueError; asset data determines the duration."""
        raise ValueError(
            f"Arb waveform length is asset-derived; data cannot be changed through set_param: {name}={value}"
        )


type RaiseWaveformCfg = Annotated[
    CosineWaveformCfg | GaussWaveformCfg | DragWaveformCfg | ArbWaveformCfg,
    Field(discriminator="style"),
]
type RaiseWaveformSpec = RaiseWaveformCfg


class _FlatTopFields(ConfigBase):
    """Shared flat-top knobs; style selects flat_top.

    length is the total duration in µs, including the rising and falling ramps.
    The flat span is length - raise_waveform.length.
    """

    style: Literal["flat_top"] = "flat_top"
    length: float | QickParam


class FlatTopWaveformSpec(_FlatTopFields):
    """Persistent flat-top; raise_waveform is a concrete key or inline ramp Spec.

    The ramp accepts cosine, gauss, drag, or arb, never const or flat_top.
    """

    raise_waveform: str | RaiseWaveformSpec


class FlatTopWaveformCfg(_FlatTopFields, AbsWaveformCfg):
    """Materialized flat-top; raise_waveform is a validated non-flat ramp Cfg."""

    raise_waveform: RaiseWaveformCfg

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set total length in µs, including rise and fall; sweeps are accepted.

        The flat span is length - raise_waveform.length. Other names raise ValueError.
        """
        if name == "length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


type WaveformSpec = Annotated[
    ConstWaveformCfg
    | CosineWaveformCfg
    | GaussWaveformCfg
    | DragWaveformCfg
    | ArbWaveformCfg
    | FlatTopWaveformSpec,
    Field(discriminator="style"),
]
type WaveformCfg = Annotated[
    ConstWaveformCfg
    | CosineWaveformCfg
    | GaussWaveformCfg
    | DragWaveformCfg
    | ArbWaveformCfg
    | FlatTopWaveformCfg,
    Field(discriminator="style"),
]
