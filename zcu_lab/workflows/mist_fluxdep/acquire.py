"""mist_fluxdep acquire helpers for lab workflows."""

from dataclasses import dataclass
from datetime import UTC, datetime

from zcu_tools.experiment.acquisition import (
    AcquisitionEvidence,
    capture_software_provenance,
    new_run_id,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.workflows import Run
from zcu_tools.program.v2 import Pulse, Readout, Reset, sweep2param

from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepCfg, PowerDepResult


@dataclass(frozen=True)
class MISTAcquisition(PowerDepResult):
    """Retain raw MIST gain response and evidence for analysis without refitting.

    gains is the actual rounded dimensionless gain vector. signals is complex
    IQ with the same shape. evidence contains fixed UTC acquisition boundaries,
    software provenance and the Run snapshot, not derived fit/threshold facts.
    """

    evidence: AcquisitionEvidence


def run_mist(run: Run[PowerDepCfg]) -> MISTAcquisition:
    """Acquire a disturbance gain trace to resolve MIST after calibrated pi preparation.

    Supply native PowerDepCfg: init_pulse is required here and represents the
    calibrated pi pulse; probe_pulse is the fixed MIST disturbance waveform.
    Native optional reset retains its meaning. gain SweepCfg is fixed by plan.
    Return actual rounded gains and complex IQ/evidence with partial data on
    cancellation. Missing preparation raises ValueError before acquisition;
    native capability/compile/I/O errors propagate and Schedule failure remains
    in Run.outcome. No fitting, device setup, plotting or saving occurs.
    """
    cfg = run.cfg
    if cfg.modules.init_pulse is None:
        raise ValueError("MIST acquisition requires a calibrated pi preparation")
    soc, soccfg = run.soc, run.soccfg
    gains = sweep2array(
        cfg.sweep.gain, "gain", {"soccfg": soccfg, "gen_ch": cfg.modules.probe_pulse.ch}
    )
    buffer = run.buffer((len(gains),), axes=(gains,))
    provenance = capture_software_provenance(context=None)
    started_at = datetime.now(UTC)
    run_id = new_run_id(started_at)
    with run.schedule(buffer) as sched:
        modules = sched.cfg.modules
        gain_sweep = sched.cfg.sweep.gain
        modules.probe_pulse.set_param("gain", sweep2param("gain", gain_sweep))
        (
            sched.prog_builder(soc, soccfg)
            .add(
                Reset("reset", modules.reset),
                Pulse("pi_pulse", modules.init_pulse),
                Pulse("mist_pulse", modules.probe_pulse),
                Readout("readout", modules.readout),
            )
            .declare_sweep("gain", gain_sweep)
            .build_and_acquire()
        )
    evidence = AcquisitionEvidence(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        completion="stopped" if run.cancel_signal.is_set() else "complete",
        provenance=provenance,
        device_snapshot=run.device_snapshot,
    )
    return MISTAcquisition(gains, buffer.array, evidence)
