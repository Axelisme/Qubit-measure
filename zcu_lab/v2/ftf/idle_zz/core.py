"""Conditional Ramsey acquisition with thermal spectator I/Xπ branches."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Delay, ProgramV2Cfg, Pulse, PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg

from zcu_lab.v2._support.device import set_flux

from .._readout import JointRoles, collect_joint_shots
from ._analysis import IdleZZAnalysis, fit_branch, plot_analysis


class IdleModules(ConfigBase):
    """Calibrated π and π/2 pulses on each role's charge generator; full mux."""

    x180_f1: PulseCfg
    x180_f2: PulseCfg
    x90_f1: PulseCfg
    x90_f2: PulseCfg
    readout: MuxReadoutCfg


class IdleZZCfg(ProgramV2Cfg, ExpCfgModel):
    """Thermal spectator branches, target Ramsey, then exactly one full mux read.

    target selects the role doing π/2→delay→phase projection. The other role
    receives I (branch 0) or Xπ (branch 1), without heralding/post-selection.
    roles are fixed capture names and original tone positions. Delay is µs,
    pulse frequency MHz, phase degrees. detune (MHz) is a required operator input:
    second π/2 phase = first.phase + 360*detune*actual_delay. Its magnitude must
    comfortably exceed expected |ζ| plus qubit frequency error so both nominal
    branches remain on its sign-selected side. No automatic detune adjustment
    occurs. dc_device=None leaves all sources untouched;
    otherwise only the named source is swept, with no compensation. Host rounds
    retain original shot order; the program copy alone uses rounds=1.
    """

    modules: IdleModules
    roles: JointRoles
    target: Literal["F1", "F2"] = "F1"
    detune: Annotated[
        float,
        Field(allow_inf_nan=False),
        FieldMeta(
            unit="MHz",
            description="Signed artificial phase ramp; magnitude must exceed expected |ζ| plus frequency error.",
        ),
    ]
    delay: Annotated[SweepCfg, FieldMeta(unit="us")]
    dc_device: Annotated[
        str | None,
        FieldMeta(
            label="Coupler DC source",
            description="Direct device name; None disables the DC sweep.",
        ),
    ] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"], FieldMeta(label="DC source working unit")
    ] = "native"
    dc_values: Annotated[
        SweepCfg,
        FieldMeta(
            label="DC source values",
            description="Inactive when source is None; no conversion or F1/F2 compensation.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require complete tones and consistent independent calibrated drive lines."""
        m = self.modules
        if m.x180_f1.ch == m.x180_f2.ch:
            raise ValueError("F1/F2 require independent charge generators")
        if m.x90_f1.ch != m.x180_f1.ch or m.x90_f2.ch != m.x180_f2.ch:
            raise ValueError("Each role's Xπ and Xπ/2 must use the same generator")
        self.roles.validate_readout(m.readout)
        delays = sweep2array(self.delay)
        if not np.all(np.isfinite(delays)) or np.any(delays < 0):
            raise ValueError("Ramsey delay must be finite and nonnegative")
        if self.dc_device is not None:
            if not self.dc_device.strip():
                raise ValueError("DC source must name a connected device or be None")
            if not np.all(np.isfinite(sweep2array(self.dc_values))):
                raise ValueError("DC source coordinates must be finite")
        return self


@dataclass(frozen=True)
class IdleZZResult:
    """Raw complex IQ (D,2,T,rounds*reps,N), original full-tone/shot order.

    dc_values has D device-native coordinates; [0] is an inactive-axis sentinel
    only when saved cfg.dc_device is None, not a measured or set DC value. branches is
    [0,1] = spectator I/Xπ, delays contains T actual quantized µs coordinates,
    shot_indices is round-major [0..rounds*reps-1], tone_indices is [0..N-1].
    Each acquired round contains all reps/tones and is finite. Cancelled whole
    round slots remain complex NaN; these are not acquired samples. Raw units
    are collect_raw_iq ADC units/sample, with offsets and thermal content intact.
    """

    dc_values: NDArray[np.float64]
    branches: NDArray[np.int64]
    delays: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


def _source_rounds(
    source: RunRecord[IdleZZCfg, IdleZZResult],
) -> tuple[IdleZZCfg, NDArray[np.bool_]]:
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError(
            "Idle ZZ analysis requires saved cfg; current Form/library is not a substitute"
        )
    ntones, nshots = len(cfg.modules.readout.tones), cfg.rounds * cfg.reps
    dc_values = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
    ndelay = cfg.delay.expts
    if (
        not np.array_equal(data.dc_values, dc_values)
        or not np.array_equal(data.branches, np.arange(2))
        or not np.array_equal(data.tone_indices, np.arange(ntones))
        or not np.array_equal(data.shot_indices, np.arange(nshots))
        or data.delays.shape != (ndelay,)
        or not np.all(np.isfinite(data.delays))
        or np.any(data.delays < 0)
        or len(np.unique(data.delays)) != ndelay
        or data.signals.shape != (len(dc_values), 2, ndelay, nshots, ntones)
    ):
        raise ValueError(
            "Idle raw axes/shape must match saved cfg: (DC,2,delay,rounds*reps,full tones)"
        )
    rounds = data.signals.reshape(
        len(dc_values), 2, ndelay, cfg.rounds, cfg.reps, ntones
    )
    acquired = np.all(np.isfinite(rounds), axis=(-2, -1))
    missing = np.all(
        np.isnan(rounds) & ~np.isinf(rounds.real) & ~np.isinf(rounds.imag),
        axis=(-2, -1),
    )
    if not np.all(acquired | missing):
        raise ValueError(
            "Idle raw contains nonfinite acquired IQ or a partially missing round"
        )
    return cfg, acquired


class IdleZZExp(PersistableExperiment[IdleZZResult, IdleZZCfg]):
    """Compare target Ramsey fringes for thermal spectator I/Xπ preparations."""

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("tone_indices", "Tone", "", IDENTITY, dtype=np.int64),
            Axis("shot_indices", "Shot", "", IDENTITY, dtype=np.int64),
            Axis("delays", "Delay", "s", US_TO_S),
            Axis("branches", "Spectator branch", "", IDENTITY, dtype=np.int64),
            Axis(
                "dc_values",
                "DC source value (inactive if dc_device=null)",
                "a.u.",
                IDENTITY,
            ),
        ),
        z=ZSpec("signals", "Raw IQ", "a.u."),
        result_type=IdleZZResult,
        cfg_type=IdleZZCfg,
        cfg_schema_version="1.0",
        tag="ftf/idle_zz",
    )

    def run(self, config: IdleZZCfg, *, context: RunContext) -> IdleZZResult:
        """Measure conditional Ramsey fringes, optionally sweeping a named DC source.

        Apply I or Xπ to the thermal spectator, then target π/2→delay→π/2
        with projection phase advanced by +360*detune*delay, followed by one
        full mux read. No reset, heralding or post-selection purifies either
        branch. Return actual delays in us and raw IQ ordered
        (DC, branch, delay, rounds*reps, full tones), not fitted ζ. Analyze
        estimates the nominal-branch frequency difference under its sign and
        single-frequency assumptions, not a purified ZZ coupling.

        Only the optional named DC source is swept; cfg is unchanged. Completed
        rounds survive cancellation, missing rounds stay NaN; the caller checks
        cancel_signal.raise_if_error() to distinguish failure from stop.
        """
        # Resolve delay/DC axes; retain each host round as raw shots, not a mean.
        cfg = config.model_copy(deep=True, update={"rounds": 1})
        delays = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
        dc_values = (
            np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
        )
        ntones = len(cfg.modules.readout.tones)
        viewer = context.plots.liveplot_1d(
            "measurement",
            "Delay (us)",
            "Completed shots (all DC/branches)",
            num_lines=1,
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            acquired = np.all(np.isfinite(raw), axis=-1).sum(axis=(0, 1, 3, 4))
            viewer.update(delays, acquired)

        buffer = SignalBuffer(
            (len(dc_values), 2, len(delays), config.rounds, cfg.reps, ntones),
            on_update=update_view,
        )

        # Sweep only the named DC source; a disabled source leaves devices untouched.
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            for dc_value, dc_step in sched.scan("dc", dc_values.tolist()):
                if cfg.dc_device is not None:
                    set_flux(
                        context.devices[cfg.dc_device],
                        dc_value,
                        unit=cfg.dc_unit,
                        stop=context.cancel_signal,
                    )
                for branch, branch_step in dc_step.scan("branch", [0, 1]):
                    for delay, step in branch_step.scan("delay", delays.tolist()):
                        builder = step.prog_builder(context.soc, context.soccfg)
                        m = cfg.modules
                        spectator = m.x180_f2 if cfg.target == "F1" else m.x180_f1
                        pi2 = m.x90_f1 if cfg.target == "F1" else m.x90_f2

                        # Compare thermal I/Xπ preparations with the same target Ramsey.
                        if branch:
                            builder.add_pulse("spectator_x180", spectator)
                        builder.add(
                            Pulse("target_pi2", pi2),
                            Delay("ramsey_delay", delay),
                            Pulse(
                                "projection",
                                pi2.with_updates(
                                    phase=pi2.phase + 360 * cfg.detune * delay
                                ),
                            ),
                            MuxReadout("readout", m.readout),
                        )

                        # Reuse this sequence while collecting full-tone shots per round.
                        prog = builder.build()
                        for _, round_step in step.repeat("round", config.rounds):
                            _ = round_step.prog_builder(
                                context.soc, context.soccfg
                            ).run_program(
                                prog, raw2signal_fn=partial(collect_joint_shots, prog)
                            )

        # Flatten rounds into the shot axis without changing tone or shot order.
        buffer.trigger_update(flush=True)
        return IdleZZResult(
            dc_values,
            np.arange(2, dtype=np.int64),
            delays,
            np.arange(config.rounds * cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array.reshape(
                len(dc_values), 2, len(delays), config.rounds * cfg.reps, ntones
            ),
        )

    def analyze(
        self,
        source: RunRecord[IdleZZCfg, IdleZZResult],
        options: None,
        *,
        plots: Plots,
    ) -> IdleZZAnalysis:
        """Estimate ζ (MHz) as the fitted Xπ-minus-I nominal-branch frequency.

        Fit the target Ramsey IQ for each complete branch at each DC setting
        using only the saved cfg and raw data.

        Second π/2 phase advances by +360*detune*delay degrees. The existing
        free-phase decay cosine gives a nonnegative fringe magnitude m (MHz).
        With operator assurance that both branches stay on the detune-selected
        side, f=sign(detune)*m and ζ=f_Xπ−f_I. The source detune must comfortably
        exceed expected |ζ| plus qubit frequency error; data do not prove this
        assumption. For negative detune, compare m to 0 and 2*abs(detune), not a
        positive magnitude to negative 2*detune. Distance <=max(1/actual delay
        span in µs, frequency covariance 1σ in MHz) is sign-unidentifiable.
        Require >=6 distinct regular delays, positive span and finite error.
        Incomplete necessary branches never produce ζ. Numeric fit failures
        retain reasons/curves; malformed raw/coordinates or missing saved cfg
        raise ValueError, never use current Form/library. No heralding, pure-ZZ
        claim, simulator truth, writeback or automatic detune/DC adjustment.
        Finite fits produce the two nominal branches' single-frequency-model
        fitted difference with fit_quality, not proof that thermal mixture was
        excluded. No automatic lack-of-fit/statistical gate is applied; the
        operator judges model applicability from residuals, errors and plots.
        """
        del options
        cfg, acquired = _source_rounds(source)
        data = source.result
        tone = cfg.roles.f1_tone if cfg.target == "F1" else cfg.roles.f2_tone
        fits = []
        zetas: list[float | None] = []
        errors: list[float | None] = []
        reasons: list[str | None] = []
        for dc_index in range(len(data.dc_values)):
            pair = [
                fit_branch(
                    dc_index=dc_index,
                    branch=branch,
                    delays=data.delays,
                    shots=data.signals[dc_index, branch, :, :, tone].reshape(
                        len(data.delays), cfg.rounds, cfg.reps
                    ),
                    acquired=acquired[dc_index, branch],
                    detune=cfg.detune,
                )
                for branch in range(2)
            ]
            fits.extend(pair)
            a, b = pair
            if (
                a.signed_mhz is None
                or b.signed_mhz is None
                or a.error_mhz is None
                or b.error_mhz is None
            ):
                zetas.append(None)
                errors.append(None)
                reasons.append(
                    "; ".join(
                        f"{label}: {fit.reason}"
                        for label, fit in zip(("I", "Xπ"), pair, strict=True)
                        if fit.reason
                    )
                )
            else:
                zetas.append(b.signed_mhz - a.signed_mhz)
                errors.append(float(np.hypot(a.error_mhz, b.error_mhz)))
                reasons.append(None)
        answer = IdleZZAnalysis(
            tuple(fits), tuple(zetas), tuple(errors), tuple(reasons)
        )
        dc_label = (
            "DC not swept (inactive coordinate)"
            if cfg.dc_device is None
            else f"{cfg.dc_device} ({cfg.dc_unit})"
        )
        plot_analysis(
            answer,
            data.delays,
            data.dc_values,
            title=f"Target {cfg.target}, detune={cfg.detune:g} MHz",
            dc_label=dc_label,
            plots=plots,
        )
        return answer
