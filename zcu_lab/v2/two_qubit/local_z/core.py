"""Thermal B00 local-Z acquisition on both captured qubit roles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import IDENTITY, AxesSpec, Axis, PersistableExperiment, ZSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg, PulseCfg
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg

from .._drive import DualLineCfg, dual_line_drive
from .._readout import JointRoles, collect_joint_shots
from ..joint_readout.core import GECenters
from .analysis import LocalZAnalysis, analyze_local_z


class LocalZModules(ConfigBase):
    """Captured X90 pulses for F1/F2 and the complete mux readout.

    x90_f1/x90_f2 prepare and project their own role; no spectator Xπ is played.
    readout retains every tone in its captured order.
    """

    x90_f1: Annotated[
        PulseCfg,
        FieldMeta(
            label="F1 Xπ/2",
            description="填入 F1 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain normalized 與 duration us。",
        ),
    ]
    x90_f2: Annotated[
        PulseCfg,
        FieldMeta(
            label="F2 Xπ/2",
            description="填入 F2 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain normalized 與 duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot。先用 lookback 取得 timing，再做 mux onetone，確認 frequency MHz、gain normalized 與 timing us。",
        ),
    ]


class LocalZCfg(ProgramV2Cfg, ExpCfgModel):
    """B00 → target X90 → n parallel CZ pairs → independent X/Y projection → mux.

    Both targets are acquired in one run, F1 then F2. The other qubit is not
    flipped. gate_counts are at least three consecutive increasing nonnegative
    integers. All tones and round-major shots persist. GE centers are optional
    saved analysis calibration.
    The two ordinary CZ pulses retain their constructive relative phase.
    """

    modules: LocalZModules
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses。先做 conditional spectroscopy／chevron，再確認 coupler 的 cz_f1／cz_f2 slots。",
        ),
    ]
    gate_counts: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4], min_length=3
    )
    f1_centers: Annotated[
        GECenters | None,
        FieldMeta(
            label="F1 GE centers",
            description="填入 singleshot/ge 取得的 F1 ground／excited complex IQ centers，使用 normalized raw ADC units。",
        ),
    ] = None
    f2_centers: Annotated[
        GECenters | None,
        FieldMeta(
            label="F2 GE centers",
            description="填入 singleshot/ge 取得的 F2 ground／excited complex IQ centers，使用 normalized raw ADC units。",
        ),
    ] = None

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require full readout, correct role generators and consecutive train lengths."""
        self.roles.validate_readout(self.modules.readout)
        for drive, x90 in (
            (self.drive.f1, self.modules.x90_f1),
            (self.drive.f2, self.modules.x90_f2),
        ):
            if drive.ch != x90.ch:
                raise ValueError("CZ and X90 must use the same captured role generator")
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        if any(b - a != 1 for a, b in pairwise(self.gate_counts)):
            raise ValueError("Gate counts must be consecutive and strictly increasing")
        return self


@dataclass(frozen=True)
class LocalZResult:
    """Raw IQ (2 targets,2 projections,Nn,rounds*reps,Ntones).

    targets=[0,1] mean F1/F2; quadratures=[0,1] mean independent X90/Y90
    projection pulse axes. gate_counts is the acquired integer train axis;
    shot_indices is round-major, tone_indices is the complete original order.
    signals is normalized ADC IQ, without heralding, filtering or averaging.
    Unacquired whole-round slots remain complex NaN; acquired IQ must be finite.
    """

    targets: NDArray[np.int64]
    quadratures: NDArray[np.int64]
    gate_counts: NDArray[np.int64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class LocalZExp(PersistableExperiment[LocalZResult, LocalZCfg]):
    """Stateless two-target local-Z acquisition, saved-source analysis and native IO."""

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("tone_indices", "Tone", "", IDENTITY, dtype=np.int64),
            Axis("shot_indices", "Shot", "", IDENTITY, dtype=np.int64),
            Axis("gate_counts", "CZ gate count", "", IDENTITY, dtype=np.int64),
            Axis("quadratures", "Projection X or Y", "", IDENTITY, dtype=np.int64),
            Axis("targets", "Target F1 or F2", "", IDENTITY, dtype=np.int64),
        ),
        z=ZSpec("signals", "Raw IQ", "a.u."),
        result_type=LocalZResult,
        cfg_type=LocalZCfg,
        cfg_schema_version="1.0",
        tag="two_qubit/local_z",
    )

    def run(self, config: LocalZCfg, *, context: RunContext) -> LocalZResult:
        """Acquire both targets in fixed projection/n/round slots without cfg mutation.

        Caller owns transport/plots and checks cancel_signal.raise_if_error() to
        distinguish transport errors from cancellation. Final raw conversion
        overwrites the same round slot, never appending duplicate shots.
        """
        cfg = config.model_copy(deep=True, update={"rounds": 1})
        ntones = len(cfg.modules.readout.tones)
        viewer = context.plots.liveplot_1d(
            "measurement",
            "CZ gate count",
            "Completed shots (both targets/projections)",
            num_lines=1,
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            viewer.update(
                np.array(cfg.gate_counts),
                np.all(np.isfinite(raw), axis=-1).sum(axis=(0, 1, 3, 4)),
            )

        buffer = SignalBuffer(
            (2, 2, len(cfg.gate_counts), config.rounds, cfg.reps, ntones),
            on_update=update_view,
        )
        m = cfg.modules
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for target_index, target_step in sched.scan("target", [0, 1]):
                target = (m.x90_f1, m.x90_f2)[target_index]
                for quadrature, projection_step in target_step.scan(
                    "quadrature", [0, 1]
                ):
                    for n, step in projection_step.scan("gate_count", cfg.gate_counts):
                        builder = step.prog_builder(context.soc, context.soccfg)
                        builder.add_pulse("prepare_target", target)
                        for gate in range(n):
                            builder.add(
                                dual_line_drive(
                                    cfg.drive,
                                    frequency=float(cfg.drive.f1.freq),
                                    name=f"cz_{gate}",
                                )
                            )
                        builder.add_pulse(
                            "projection",
                            target.with_updates(phase=target.phase + 90 * quadrature),
                        )
                        builder.add(MuxReadout("readout", m.readout))
                        prog = builder.build()
                        for _, round_step in step.repeat("round", config.rounds):
                            _ = round_step.prog_builder(
                                context.soc, context.soccfg
                            ).run_program(
                                prog,
                                raw2signal_fn=partial(collect_joint_shots, prog),
                            )
        buffer.trigger_update(flush=True)
        return LocalZResult(
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.array(cfg.gate_counts, dtype=np.int64),
            np.arange(config.rounds * cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array.reshape(
                2, 2, len(cfg.gate_counts), config.rounds * cfg.reps, ntones
            ),
        )

    def analyze(
        self,
        source: RunRecord[LocalZCfg, LocalZResult],
        options: None,
        *,
        plots: Plots,
    ) -> LocalZAnalysis:
        """Analyze saved cfg/raw only; incomplete projection pairs never imply phase."""
        del options
        return analyze_local_z(source, plots=plots)
