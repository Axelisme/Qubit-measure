"""Session value types — the SoC structural surfaces + the active context.

These are the session-core value types every measurement-session app speaks in:
the minimal ``SocHandle`` / ``SocCfgHandle`` surfaces a connected QICK board
exposes, and ``SessionEnv`` (the published parameters/soc/project bundle) + its
``ContextReadiness`` lifecycle. They carry no experiment cfg-tree coupling
(``CfgSchema``, the Spec/Value trees, ``RunRequest``/analyze/writeback types stay
in the adapter layer), so they sit below the apps in ``gui/session``. Their
session-core imports provide parameter captures and cached value lookup.

The adapter package re-exports ``SessionEnv`` / ``ContextReadiness`` /
``SocHandle`` / ``SocCfgHandle`` because ``ExpAdapterProtocol``'s own signatures
speak in them — i.e. they are part of the adapter contract's vocabulary, sourced
from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias

from zcu_tools.gui.session.parameters import ParameterContext as ParameterContext
from zcu_tools.gui.session.parameters import RoleBinding as RoleBinding
from zcu_tools.gui.session.value_lookup import EmptyValueLookup, ValueLookup


class SocProtocol(Protocol):
    """Minimal QICK SoC surface carried through GUI run requests."""

    def get_cfg(self) -> Mapping[str, object]: ...


class SocCfgProtocol(Protocol):
    """Minimal QICK config surface used by adapters and setup UI."""

    def description(self) -> str: ...
    def dump_cfg(self) -> str: ...  # QICK config as a JSON string
    def cycles2us(
        self, cycles: Any, /, gen_ch: Any = None, ro_ch: Any = None
    ) -> Any: ...
    def us2cycles(
        self,
        us: Any,
        /,
        gen_ch: Any = None,
        ro_ch: Any = None,
        # tolerated: QICK us2cycles accepts as_float=False positionally after gen_ch/ro_ch.
        as_float: bool = False,  # noqa: FBT001, FBT002
    ) -> Any: ...
    def freq2reg(self, freq: Any, /, gen_ch: Any = None, ro_ch: Any = None) -> Any: ...
    def reg2freq(self, reg: Any, /, gen_ch: Any = None) -> Any: ...
    def deg2reg(self, deg: Any, /, gen_ch: Any = None, ro_ch: Any = None) -> Any: ...
    def reg2deg(self, reg: Any, /, gen_ch: Any = None, ro_ch: Any = None) -> Any: ...
    def get_maxv(self, ch: int, /) -> Any: ...
    def __getitem__(self, key: str) -> object: ...


SocHandle: TypeAlias = SocProtocol
SocCfgHandle: TypeAlias = SocCfgProtocol


class ContextReadiness(Enum):
    """Published document scope, independent of hardware availability.

    EMPTY means no entry is open and parameters is None. GLOBAL uses the opened
    entry's documents with no point label. POINT uses a saved point's documents.
    Both GLOBAL and POINT provide a file-backed parameter environment.
    """

    EMPTY = "empty"
    GLOBAL = "global"
    POINT = "point"


@dataclass(frozen=True)
class SessionEnv:
    """Published parameters plus independent live session capabilities.

    parameters is None for EMPTY, otherwise a fixed capture without consumer
    roles targeting SavedEntry or SavedPoint. soc/soccfg remain live capabilities,
    not Run evidence. result_dir/database_path cache opened entry paths, empty
    without an entry. active_label is the saved point label in POINT and None in
    EMPTY/GLOBAL. readiness distinguishes those three scopes; GLOBAL and POINT
    both provide a file-backed context. values reads published value-source cache.
    Retaining an environment does not update its parameter capture.
    """

    parameters: ParameterContext | None
    soc: SocHandle | None
    soccfg: SocCfgHandle | None
    result_dir: str = ""
    database_path: str = ""
    active_label: str | None = None
    readiness: ContextReadiness = ContextReadiness.EMPTY
    values: ValueLookup = field(default_factory=EmptyValueLookup)

    # -- readiness predicates (the context answers about itself) -----------

    def has_context(self) -> bool:
        """Return whether an entry-level or point-level environment is installed."""
        return self.readiness is not ContextReadiness.EMPTY

    def is_active(self) -> bool:
        """File-backed context eligible for run and save."""
        return self.readiness in (ContextReadiness.GLOBAL, ContextReadiness.POINT)

    def has_soc(self) -> bool:
        return self.soc is not None
