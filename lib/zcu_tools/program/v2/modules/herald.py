"""Host post-selection resets that reuse the terminal readout declaration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator, Field
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import (
    CircleClassifier,
    ConfigBase,
    HeraldState,
    LinearClassifier,
    MuxHeraldKeep,
)

from ..macro.readout import ReadoutEmission
from ..readout_observation import ReadoutPath, observe_readout_paths
from .base import AbsModuleCfg, resolve_module_ref
from .pulse import PulseCfg
from .readout import (
    MuxReadout,
    MuxReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    classification_processor,
)
from .reset import AbsReset, ResetCfg

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ReadoutClassifier

    from ...acquisition import ShotMaskSource
    from ..timeline import TimelineProgramV2


type _PrepareCfg = Annotated[
    Annotated[PulseCfg | ResetCfg, Field(discriminator="type")],
    BeforeValidator(resolve_module_ref),
]


class _HeraldResetCfg(AbsModuleCfg):
    prepare: _PrepareCfg | None = None

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Reject every sweep knob; the reset has no tunable playback parameters."""
        raise ValueError(f"{self.type} does not support parameter {name!r}={value!r}")


class HeraldResetCfg(_HeraldResetCfg):
    """Pulse-readout herald: keep g/e/other; prepare is one optional pulse or ordinary reset.

    readout must match the terminal pulse readout. Its delays also apply to
    the herald read. Core materialization resolves inline settings and keys
    before converting to this native cfg.
    """

    type: Literal["reset/herald"] = "reset/herald"
    readout: Annotated[PulseReadoutCfg, BeforeValidator(resolve_module_ref)]
    keep: HeraldState

    def build(
        self, name: str, *, terminal: PulseReadout | MuxReadout | None = None
    ) -> HeraldReset:
        """Capture these settings for emit; terminal is the same readout object sent to emit later.

        When terminal is omitted, emit the returned reset's readout as the
        terminal readout. Program build rejects a mismatch, sweeps, and missing
        or incompatible calibration.
        """
        return HeraldReset(name, self, terminal=terminal)


class MuxHeraldResetCfg(_HeraldResetCfg):
    """Keep shots matching every selected tone of the shared mux readout.

    keep maps positions in the full tones list to g, e or other; decimal
    string keys convert to integer positions. Unlisted tones still acquire.
    prepare is one optional pulse or ordinary reset, played first.
    """

    type: Literal["reset/mux_herald"] = "reset/mux_herald"
    readout: Annotated[MuxReadoutCfg, BeforeValidator(resolve_module_ref)]
    keep: MuxHeraldKeep

    def build(
        self, name: str, *, terminal: PulseReadout | MuxReadout | None = None
    ) -> HeraldReset:
        """Capture these settings for emit; terminal is the same mux readout object emitted later.

        When omitted, emit the returned reset's readout as the terminal.
        Program build rejects a mismatch, invalid tone positions, sweeps, and
        missing or incompatible calibration for selected tones.
        """
        return HeraldReset(name, self, terminal=terminal)


ResetWithHeraldCfg: TypeAlias = Annotated[
    ResetCfg | HeraldResetCfg | MuxHeraldResetCfg, Field(discriminator="type")
]


class HeraldReset(AbsReset):
    """Emit optional preparation and one hidden read before the caller's sequence."""

    def __init__(
        self,
        name: str,
        cfg: HeraldResetCfg | MuxHeraldResetCfg,
        *,
        terminal: PulseReadout | MuxReadout | None = None,
    ) -> None:
        """Snapshot cfg; if supplied, borrow terminal for both readout emissions.

        name labels errors and the preparation. Otherwise, create the readout
        from cfg and expose it as readout so the caller can emit it at the end.
        Emit this module before the experiment and its readout after it.
        Configuration and read placement errors surface at program build.
        """
        self.name = name
        self._cfg = deepcopy(cfg)
        self.readout = (
            self._cfg.readout.build(f"{name}_readout") if terminal is None else terminal
        )
        prepare = self._cfg.prepare
        self._prepare = None if prepare is None else prepare.build(f"{name}_prepare")

    def declare(self, prog: TimelineProgramV2) -> None:
        """Validate terminal readout settings, then declare preparation and the shared readout."""
        configured = _fixed_settings(self._cfg.readout, f"{self.name}.readout")
        actual = _fixed_settings(self.readout.cfg, f"{self.name}.terminal")
        if configured != actual:
            raise ValueError(
                f"{self.name}: herald readout differs from terminal readout"
            )
        if self._prepare is not None:
            prog.declare(self._prepare)
        prog.declare(self.readout)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Play prepare then the shared readout on timeline; tag labels the hidden emission."""
        if self._prepare is not None:
            prog.emit(self._prepare, timeline=timeline)
        channels = self._channels()
        prog.append_macro(_HeraldEmission(self, channels[0], tag, self._mask()))
        prog.emit(self.readout, timeline=timeline, tag=tag)

    def register_shot_mask_sources(self, prog: TimelineProgramV2) -> None:
        """Register each herald emission's fixed physical reads after body assembly.

        Every feasible path must consume the same reads and end with the shared
        terminal owner. Invalid placement raises ValueError before acquisition.
        """
        paths = iter(observe_readout_paths(prog))
        reads = _path_herald_reads(next(paths), self)
        for path in paths:
            if _path_herald_reads(path, self) != reads:
                raise ValueError(
                    "herald readout must consume the same reads on every path"
                )
        for marker, channels in reads.values():
            prog.register_shot_mask_source(channels, marker.mask)

    def _channels(self) -> list[int]:
        cfg = self._cfg.readout
        return (
            [cfg.ro_ch]
            if isinstance(cfg, PulseReadoutCfg)
            else [tone.ro_ch for tone in cfg.tones]
        )

    def _mask(self) -> ShotMaskSource:
        selections = _selections(self.name, self._cfg)
        first_adc = self._channels()[0]

        def keep(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.bool_]:
            kept = np.ones(iq[first_adc].shape[:2], dtype=np.bool_)
            for ch, matches in selections.items():
                kept &= matches(iq[ch][..., 0, :])
            return kept

        return keep


type _MatchesState = Callable[[NDArray[np.float64]], NDArray[np.bool_]]


def _selections(
    name: str, cfg: HeraldResetCfg | MuxHeraldResetCfg
) -> dict[int, _MatchesState]:
    where = f"{name}.readout"
    if isinstance(cfg, HeraldResetCfg):
        return {
            cfg.readout.ro_ch: _classifier_selection(
                where, cfg.readout.classifier, cfg.keep
            )
        }
    selected: dict[int, _MatchesState] = {}
    for index, state in cfg.keep.items():
        if index >= len(cfg.readout.tones):
            raise ValueError(f"{where}: keep tone {index} does not exist")
        tone = cfg.readout.tones[index]
        selected[tone.ro_ch] = _classifier_selection(
            f"{where} tone {index}", tone.classifier, state
        )
    return selected


def _fixed_settings(value: object, path: str) -> object:
    if isinstance(value, QickParam):
        if value.is_sweep():
            raise ValueError(f"{path}: herald readout does not support sweeps")
        return value.start
    if isinstance(value, ConfigBase):
        return {name: _fixed_settings(field, f"{path}.{name}") for name, field in value}
    return value


def _classifier_selection(
    where: str, classifier: ReadoutClassifier | None, state: HeraldState
) -> _MatchesState:
    if classifier is None:
        raise ValueError(f"{where}: classifier missing")
    if isinstance(classifier, CircleClassifier):
        if classifier.radius is None:
            raise ValueError(f"{where}: circle radius missing; run GE Post analysis")
        states = ("g", "e", "other")
    elif isinstance(classifier, LinearClassifier):
        states = ("g", "e")
    else:
        raise ValueError(
            f"{where}: {classifier.mode} classifier is unsupported for herald"
        )
    if state not in states:
        raise ValueError(
            f"{where}: keep state {state!r} is not part of the {classifier.mode} classifier"
        )
    classify = classification_processor(classifier)
    index = states.index(state)

    def matches(iq: NDArray[np.float64]) -> NDArray[np.bool_]:
        return classify(iq)[..., index] == 1

    return matches


class _HeraldEmission(ReadoutEmission):
    """Mark the following shared readout emission as hidden."""

    def __init__(
        self, herald: HeraldReset, first_adc: int, tag: str | None, mask: ShotMaskSource
    ) -> None:
        super().__init__(herald, first_adc, tag)
        self.mask = mask


type _HeraldReads = dict[int, tuple[_HeraldEmission, dict[int, tuple[int, ...]]]]


def _path_herald_reads(path: ReadoutPath, herald: HeraldReset) -> _HeraldReads:
    """Find each herald's physical trigger indices and check the visible terminal owner."""
    counts: dict[int, int] = {}
    controlled: set[int] = set()
    selected: _HeraldReads = {}
    pending: _HeraldEmission | None = None
    terminal: object | None = None
    for observation in path.observations:
        source = observation.source
        if isinstance(source, ReadoutEmission):
            if isinstance(source, _HeraldEmission):
                pending = source
                terminal = None
            elif pending is None:
                terminal = source.owner
        else:
            if observation.control_scope is not None:
                controlled.update(source.ros)
            if pending is not None:
                if controlled.intersection(source.ros):
                    raise ValueError(
                        "herald readout requires fixed physical read indices"
                    )
                if pending.owner is herald:
                    selected[id(pending)] = (
                        pending,
                        {ch: (counts.get(ch, 0),) for ch in source.ros},
                    )
                pending = None
            for ch in source.ros:
                counts[ch] = counts.get(ch, 0) + 1
    if selected and terminal is not herald.readout:
        raise ValueError(
            f"{herald.name}: emit the shared herald readout as the terminal readout"
        )
    return selected
