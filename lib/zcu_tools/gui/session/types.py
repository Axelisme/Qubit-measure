"""Session value types — the SoC structural surfaces + the active context.

These are the session-core value types every measurement-session app speaks in:
the minimal ``SocHandle`` / ``SocCfgHandle`` surfaces a connected QICK board
exposes, and ``ExpContext`` (the active md/ml/soc/project bundle) + its
``ContextReadiness`` lifecycle. They carry no experiment cfg-tree coupling
(``CfgSchema``, the Spec/Value trees, ``RunRequest``/analyze/writeback types stay
in the adapter layer), so they sit below the apps in ``gui/session`` and are
import-clean (only TYPE_CHECKING references to MetaDict/ModuleLibrary/predictor).

The adapter package re-exports ``ExpContext`` / ``ContextReadiness`` /
``SocHandle`` / ``SocCfgHandle`` because ``ExpAdapterProtocol``'s own signatures
speak in them — i.e. they are part of the adapter contract's vocabulary, sourced
from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, TypeAlias

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.meta_tool.metadict import MetaDict
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor


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
        as_float: bool = False,
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
    """Lifecycle readiness for operations that need a persisted context."""

    EMPTY = "empty"
    DRAFT = "draft"
    ACTIVE = "active"


@dataclass(frozen=True)
class ExpContext:
    md: MetaDict
    ml: ModuleLibrary
    soc: SocHandle | None
    soccfg: SocCfgHandle | None
    chip_name: str = "unknown_chip"
    qub_name: str = "unknown_qubit"
    res_name: str = "unknown_resonator"
    result_dir: str = ""
    database_path: str = ""
    active_label: str = ""
    predictor: FluxoniumPredictor | None = None
    readiness: ContextReadiness = ContextReadiness.EMPTY

    # -- readiness predicates (the context answers about itself) -----------

    def has_context(self) -> bool:
        """Any valid context exists (startup DRAFT or file-backed ACTIVE)."""
        return self.readiness is not ContextReadiness.EMPTY

    def is_draft(self) -> bool:
        return self.readiness is ContextReadiness.DRAFT

    def is_active(self) -> bool:
        """File-backed context eligible for run and save."""
        return self.readiness is ContextReadiness.ACTIVE

    def has_soc(self) -> bool:
        return self.soc is not None
