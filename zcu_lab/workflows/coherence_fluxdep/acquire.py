"""coherence_fluxdep acquire helpers for lab workflows."""

from dataclasses import dataclass
from datetime import UTC, datetime

from zcu_tools.experiment.acquisition import (
    AcquisitionEvidence,
    capture_software_provenance,
    new_run_id,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.workflows import Run
from zcu_tools.program.v2 import Delay, Pulse, Readout, Reset, sweep2param

from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyCfg, T2RamseyResult


@dataclass(frozen=True, kw_only=True)
class T2RamseyAcquisition(T2RamseyResult):
    """Native Ramsey IQ trace with acquisition-time evidence.

    times is the native rounded delay vector in microseconds; signals is
    complex IQ with the same shape. true_activate_detune is the actual MHz
    phase-ramp frequency after native time/phase quantization. evidence retains
    acquisition UTC boundaries, provenance and the Run device snapshot.
    """

    evidence: AcquisitionEvidence


def run_t2ramsey(run: Run[T2RamseyCfg]) -> T2RamseyAcquisition:
    """Measure Ramsey fringes to observe dephasing between two pi/2 pulses.

    Supply complete native cfg with scalar detune in MHz and length sweep in
    microseconds, plus Run soc/soccfg/device snapshot. The second pi/2 phase
    advances by 360*detune*delay. Return rounded delays, complex IQ and evidence;
    cancelled acquisition retains partial data for Engine interpretation.
    Schedule failure remains in Run.outcome. Native capability, compilation
    and acquisition errors propagate. No setup, fit, plot or save occurs.
    """
    cfg = run.cfg
    soc, soccfg = run.soc, run.soccfg
    times = sweep2array(cfg.sweep.length, "time", {"soccfg": soccfg})
    buffer = run.buffer((len(times),), axes=(times,))
    provenance = capture_software_provenance(context=None)
    started_at = datetime.now(UTC)
    run_id = new_run_id(started_at)
    # Execute the pulse sequence using schedule-local modules and sweeps.
    with run.schedule(buffer) as sched:
        modules = sched.cfg.modules
        length_sweep = cfg.sweep.length
        length_param = sweep2param("length", length_sweep)
        (
            sched.prog_builder(soc, soccfg)
            .add(
                Reset("reset", modules.reset),
                Pulse("pi2_pulse1", modules.pi2_pulse),
                Delay("t2r_delay", delay=length_param),
                Pulse(
                    "pi2_pulse2",
                    modules.pi2_pulse.with_updates(
                        phase=modules.pi2_pulse.phase + 360 * cfg.detune * length_param
                    ),
                ),
                Readout("readout", modules.readout),
            )
            .declare_sweep("length", length_sweep)
            .build_and_acquire()
        )
    # Retain acquisition identity, device evidence and partial-stop status.
    evidence = AcquisitionEvidence(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        completion="stopped" if run.cancel_signal.is_set() else "complete",
        provenance=provenance,
        device_snapshot=run.device_snapshot,
    )
    return T2RamseyAcquisition(
        times=times,
        signals=buffer.array,
        true_activate_detune=actual_detune(
            cfg.sweep.length,
            detune=cfg.detune,
            gen_ch=cfg.modules.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=1.0,
        ),
        evidence=evidence,
    )
