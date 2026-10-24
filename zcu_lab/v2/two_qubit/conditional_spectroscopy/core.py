"""Thermal four-branch, two-charge-line higher-level spectroscopy."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_flux

from .._drive import DualLineCfg, dual_line_drive
from .._readout import JointRoles, collect_joint_shots
from .analysis import SpectroscopyAnalysis, analyze_spectra

BRANCH_LABELS = ("00", "01", "10", "11")


class SpectroscopyCfg(ConfigBase):
    """B00/01/10/11 → dual weak probe → optional Xπ⊗Xπ → full mux.

    roles contains fixed captured component/tone identities. drive contains two
    ordinary scalar charge pulses with explicit waveforms. frequency is the common
    MHz host scan; drive's saved base frequency is replaced only on program copies.
    contrast applies both calibrated Xπ pulses after the probe when True, neither
    when False. No heralding or shot selection occurs. dc_device=None leaves all
    sources untouched; otherwise the direct device name is swept in dc_unit at
    dc_values, without conversion, compensation or automatic restoration.
    reps/rounds retain all shots in round-major order, not averaged IQ.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    x180_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ",
            description="填入 F1 calibrated x180 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain normalized 與 duration us。",
        ),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ",
            description="填入 F2 calibrated x180 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain normalized 與 duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot。先用 lookback 取得 timing，再做 mux onetone，確認 frequency MHz、gain normalized 與 timing us。",
        ),
    ]
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses。首次範圍由操作者確認；已有校準時可沿用 coupler 的 cz_f1／cz_f2 slots，不假設它們適合作 weak probes。",
        ),
    ]
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Frequency",
            description="填入 common probe frequency 起訖值，單位 MHz。先用 twotone frequency 定位 F1/F2，再由操作者確認高能階掃描範圍。",
        ),
    ]
    contrast: Annotated[bool, FieldMeta(label="Double Xπ contrast")] = True
    dc_device: Annotated[str | None, FieldMeta(label="Coupler DC source")] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"], FieldMeta(label="DC source working unit")
    ] = "native"
    dc_values: Annotated[
        SweepCfg,
        FieldMeta(
            label="DC source values",
            description="Inactive when source is None; no F1/F2 compensation.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require full tones, each probe on its preparation line, and finite axes."""
        self.roles.validate_readout(self.readout)
        if self.drive.f1.ch != self.x180_f1.ch or self.drive.f2.ch != self.x180_f2.ch:
            raise ValueError("Each probe must use its role's calibrated Xπ generator")
        if not np.all(np.isfinite(sweep2array(self.frequency))):
            raise ValueError("Probe frequencies must be finite")
        if self.dc_device is not None:
            if not self.dc_device.strip():
                raise ValueError("DC source must name a connected device or be None")
            if not np.all(np.isfinite(sweep2array(self.dc_values))):
                raise ValueError("DC source coordinates must be finite")
        return self


@dataclass(frozen=True)
class SpectroscopyResult:
    """Raw complex IQ (D,4,F,rounds*reps,N) with all same-shot full tones.

    dc_values are device-native coordinates, or [0] as an inactive sentinel only
    when cfg.dc_device=None. branches=[0,1,2,3] label B00/01/10/11, frequencies
    are the common requested MHz coordinates. shot_indices=[0..rounds*reps-1]
    are round-major; tone_indices=[0..N-1] retain original program tone order.
    Acquired round slots are finite ADC units/sample from collect_raw_iq.
    Cancelled unacquired whole rounds remain complex NaN, never valid samples.
    """

    dc_values: NDArray[np.float64]
    branches: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class SpectroscopyExp(Experiment[SpectroscopyResult, SpectroscopyCfg]):
    """Stateless thermal spectroscopy, diagnostic analysis and native persistence."""

    AXES_SPEC: ClassVar[AxesSpec[SpectroscopyResult, SpectroscopyCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
            AxesSpec.Axis("shot_indices", "Shot", "", dtype=np.int64),
            AxesSpec.Axis("frequencies", "Probe frequency", "Hz", 1e6),
            AxesSpec.Axis("branches", "Nominal branch", "", dtype=np.int64),
            AxesSpec.Axis(
                "dc_values", "DC source value (inactive if dc_device=null)", "a.u."
            ),
        ),
        z=AxesSpec.Z("signals", "Raw IQ", "a.u."),
        result_type=SpectroscopyResult,
        cfg_type=SpectroscopyCfg,
        cfg_schema_version="3.0",
        tag="two_qubit/conditional_spectroscopy",
    )

    def run(
        self, config: SpectroscopyCfg, *, context: RunContext
    ) -> SpectroscopyResult:
        """Acquire DC/nominal branch/frequency/host-round slots using full mux.

        Only external acquisition and the optional named DC device have effects.
        Never mutate caller cfg, read library/aliases, save or propose calibration.
        Schedule preserves completed rounds on cancel; missing slots remain NaN.
        The operation owner must check context.scope.stop.raise_if_error() so
        invalid acquired IQ or transport failures cannot masquerade as cancellation.
        """
        cfg = config.model_copy(deep=True, update={"rounds": 1})
        frequencies = sweep2array(cfg.frequency)
        dc_values = (
            np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
        )
        ntones = len(cfg.readout.tones)
        viewer = live_line(
            context.plots,
            "measurement",
            "Common probe frequency (MHz)",
            "Completed shots (all DC/branches)",
            num_lines=1,
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            acquired = np.all(np.isfinite(raw), axis=-1).sum(axis=(0, 1, 3, 4))
            viewer.update(frequencies, acquired)

        buffer = SignalBuffer(
            (len(dc_values), 4, len(frequencies), config.rounds, cfg.reps, ntones),
            on_update=update_view,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for dc_value, dc_step in sched.scan("dc", dc_values.tolist()):
                if cfg.dc_device is not None:
                    set_flux(
                        context.devices[cfg.dc_device],
                        dc_value,
                        unit=cfg.dc_unit,
                        stop=context.scope.stop,
                    )
                for branch, branch_step in dc_step.scan("branch", [0, 1, 2, 3]):
                    for frequency, step in branch_step.scan(
                        "frequency", frequencies.tolist()
                    ):
                        builder = step.prog_builder(
                            context.soc,
                            context.soccfg,
                            reps=step.cfg.reps,
                            initial_delay=step.cfg.initial_delay,
                            relax_delay=step.cfg.relax_delay,
                        )
                        m = cfg
                        if branch & 2:
                            builder.add_pulse("prepare_f1", m.x180_f1)
                        if branch & 1:
                            builder.add_pulse("prepare_f2", m.x180_f2)
                        builder.add(
                            dual_line_drive(
                                cfg.drive, frequency=frequency, name="probe"
                            )
                        )
                        if cfg.contrast:
                            builder.add_pulse("contrast_f1", m.x180_f1)
                            builder.add_pulse("contrast_f2", m.x180_f2)
                        builder.add(MuxReadout("readout", m.readout))
                        prog = builder.build()
                        for _, round_step in step.repeat("round", config.rounds):
                            _ = round_step.prog_builder(
                                context.soc,
                                context.soccfg,
                                reps=round_step.cfg.reps,
                                initial_delay=round_step.cfg.initial_delay,
                                relax_delay=round_step.cfg.relax_delay,
                            ).run_program(
                                prog,
                                raw2signal_fn=partial(collect_joint_shots, prog),
                                rounds=round_step.cfg.rounds,
                            )
        buffer.trigger_update(flush=True)
        return SpectroscopyResult(
            dc_values,
            np.arange(4, dtype=np.int64),
            frequencies,
            np.arange(config.rounds * cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array.reshape(
                len(dc_values), 4, len(frequencies), config.rounds * cfg.reps, ntones
            ),
        )

    def analyze(
        self,
        source: RunRecord[SpectroscopyCfg, SpectroscopyResult],
        options: None,
        *,
        plots: Plots,
    ) -> SpectroscopyAnalysis:
        """Plot all-tone mean IQ for complete branches from saved cfg/raw only.

        Each DC/branch requires every frequency and requested round. Incomplete
        branches report counts/reasons, not spectra from a biased partial scan.
        No peak naming, Hamiltonian fit, population inversion or writeback occurs.
        Missing saved cfg and malformed/nonfinite acquired raw raise ValueError.
        """
        del options
        return analyze_spectra(source, plots=plots)
