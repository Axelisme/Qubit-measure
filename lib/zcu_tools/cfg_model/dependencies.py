"""Typed dependency contracts shared by schema and asset owners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from zcu_tools.core import waveform_format

if TYPE_CHECKING:
    from .module import ModuleSpec
    from .waveform import WaveformSpec


class WaveformAssetInfo(Protocol):
    """Asset reads bound to one explicit scope, without caching asset bytes.

    Keys identify assets in the bound scope, not paths or GUI roles. duration
    and samples read current bytes; path_for only checks key and file existence. Missing or invalid assets and I/O failures raise an
    owner-defined exception; no empty data or duration fallback is permitted.
    Copying a consumer must not rebind this dependency to another scope.
    """

    def duration(self, data_key: str) -> float:
        """Return current asset time[-1] in microseconds; propagate read errors."""
        ...

    def samples(self, data_key: str) -> waveform_format.WaveformSamples:
        """Read isolated, validated I/Q samples and their reference time axis."""
        ...

    def path_for(self, data_key: str) -> Path:
        """Return an existing asset's absolute path in this fixed scope.

        Validate the concrete key and file existence, but do not load samples
        or recipes. Propagate missing-key/key-grammar/I/O errors. No global
        rebinding, file creation, byte capture, or fallback is permitted.
        """
        ...


class ModuleResolver(Protocol):
    """Read unexpanded Specs from one fixed library snapshot using concrete keys.

    Implementations isolate returned values and validate key-family partitions.
    Missing keys raise KeyError with key evidence; invalid roots or family
    partitions raise ValueError. The caller translates both during materialization.
    No focus, role shorthand, asset lookup, or child materialization is performed.
    """

    def module_spec(self, path: str) -> ModuleSpec:
        """Copy a module Spec; missing keys raise KeyError, invalid roots ValueError."""
        ...

    def waveform_spec(self, path: str) -> WaveformSpec:
        """Copy a waveform Spec; missing keys raise KeyError, invalid roots ValueError."""
        ...


@dataclass(frozen=True)
class CfgValidationContext:
    """Explicit dependencies for one materialization call.

    modules supplies the fixed library snapshot for string references, or None
    when no lookup is needed. assets supplies duration/samples/path reads in one
    fixed asset scope, or None when no arb envelope is needed. Missing required dependencies raise
    ValueError at use; no global fallback or caching is permitted.
    """

    modules: ModuleResolver | None = None
    assets: WaveformAssetInfo | None = None
