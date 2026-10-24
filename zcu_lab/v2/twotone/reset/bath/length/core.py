from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickSweep1D
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting import live_line
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import BathReset, Pulse, Readout, Reset, SweepCfg


def _default_phase_values() -> NDArray[np.float64]:
    return np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)


@dataclass(frozen=True)
class LengthResult:
    """Qubit-tone lengths in us and mean IQ (length, absolute pi/2 phase in degrees)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    phases: NDArray[np.float64] = dataclass_field(default_factory=_default_phase_values)


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.asarray(
        np.abs(signals[..., 2] - signals[..., 0]) ** 2
        + np.abs(signals[..., 3] - signals[..., 1]) ** 2,
        dtype=np.float64,
    )  # (lengths, )


class LengthCfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[BathReset.Cfg, FieldMeta(label="Tested reset")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="us")]


class LengthExp(Experiment[LengthResult, LengthCfg]):
    AXES_SPEC: ClassVar[AxesSpec[LengthResult, LengthCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("phases", "Pi2 Phase", "deg", dtype=np.float64),
            AxesSpec.Axis("lengths", "Length", "s", scale=1e-6, dtype=np.float64),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/bath/length",
    )

    def run(
        self,
        config: LengthCfg,
        *,
        context: RunContext,
    ) -> LengthResult:
        """Measure bath-reset tomography contrast versus qubit-tone duration.

        After optional reset and initialization, start cavity and qubit tones
        together, then apply the pi/2 pulse after the cavity tone and read
        out. Use absolute pi/2 phases of 0, 90, 180 and 270 degrees.
        Host-scan qubit-tone length in us while
        preserving the configured cavity-minus-qubit length difference.
        Return lengths and round-averaged complex IQ shaped (Nlength, 4).
        Opposite-phase contrast is not a calibrated reset fidelity.
        """
        # Prepare qubit-tone lengths while preserving the cavity-tone duration offset.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg

        rounds = cfg.rounds  # round loop is driven by Schedule.repeat below

        length_sweep = cfg.length
        tested_reset = modules.tested_reset
        qub_lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": tested_reset.qubit_tone_cfg.ch},
            allow_array=True,
        )
        lengths = qub_lengths  # use qubit tone length as x-axis

        orig_res_length = tested_reset.cavity_tone_cfg.waveform.length
        orig_qub_length = tested_reset.qubit_tone_cfg.waveform.length
        length_diff = orig_res_length - orig_qub_length

        prog_cache: dict[float, Any] = {}

        def average_signals(
            signals: NDArray[np.complex128],
        ) -> NDArray[np.complex128]:
            _signals = np.asarray(signals)  # shape: (rounds, lengths, 4)
            mean_signals = np.full_like(_signals[0], fill_value=np.nan)
            mask = np.any(np.isfinite(_signals), axis=0)
            mean_signals[mask] = np.nanmean(_signals[:, mask], axis=0)
            return mean_signals  # (lengths, 4)

        # Show tomography contrast, but retain phase-resolved IQ for round averaging.
        viewer = live_line(context.plots, "measurement", "Length (us)", "Signal (a.u.)")
        buffer = SignalBuffer(
            (rounds, len(lengths), 4),
            on_update=lambda data: viewer.update(
                lengths, bathreset_signal2real(average_signals(data))
            ),
        )

        # Repeat the host length scan; each program sweeps the four pi/2 phases.
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            length_values = lengths.tolist()
            for _, rep in sched.repeat("rounds", rounds):
                for length, step in rep.scan("length", length_values):
                    step.cfg.rounds = 1
                    modules = step.cfg
                    tested_reset = modules.tested_reset
                    tested_reset.set_param("qub_length", length)
                    tested_reset.set_param("res_length", length + length_diff)
                    tested_reset.set_param(
                        "pi2_phase", QickSweep1D("phase", 0.0, 270.0)
                    )

                    cache_key = float(tested_reset.qubit_tone_cfg.waveform.length)
                    builder = step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    program = prog_cache.get(cache_key)
                    if program is None:
                        program = (
                            builder.add(
                                Reset("reset", modules.reset),
                                Pulse("init_pulse", modules.init_pulse),
                                BathReset("tested_reset", tested_reset),
                                Readout("readout", modules.readout),
                            )
                            .declare_sweep("phase", 4)
                            .build()
                        )
                        prog_cache[cache_key] = program

                    _ = builder.run_program(program, rounds=cfg.rounds)
        signals = average_signals(buffer.array)

        return LengthResult(
            lengths=lengths,
            signals=signals,
            phases=_default_phase_values(),
        )

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot the four-phase contrast versus qubit-tone length in us.

        Use opposite-phase IQ differences from source.result and publish fit.
        No duration is fitted or returned; source.cfg may be None.
        """
        del options
        result = source.result

        lens, signals = result.lengths, result.signals

        real_signals = bathreset_signal2real(signals)

        _, ax = plots.subplots("fit")

        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("Length (us)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title("Bath Reset Length Measurement")
        ax.grid(True)
