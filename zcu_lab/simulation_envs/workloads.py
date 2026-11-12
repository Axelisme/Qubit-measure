"""Offline real-experiment workloads for numeric family diagnostics.

Providers execute the actual experiment run against an explicitly simulated
backend, then consume the public detached replay records. No hardware, GUI,
working-copy configuration or tools imports are used.
"""

import subprocess
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.probe import run_stage
from zcu_tools.program import build_soccfg
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulate.evolution import ModelSource, TimelineBatch, segment_spans
from zcu_tools.simulators.replay import read_replay, replay_paths
from zcu_tools.simulators.single_fluxonium.config import (
    SingleFluxoniumSettings,
    parse_single_fluxonium,
)
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumSource,
    single_fluxonium,
)

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqExp

_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS = "zcu_lab/simulation_envs/single_fluxonium/default.yaml"
_PROFILE = "zcu_lab/simulation_envs/profiles/qick_216_v2.yaml"
_EXPERIMENT = "zcu_lab/v2/twotone/freq/core.py"
_FORM = "zcu_lab/v2/twotone/freq/form.py"

# Candidate domains from evolution's public Kernel summary, not guard promises.
_FAMILY_DOMAINS = {
    "t4": ("idle",),
    "adiabatic_map": ("adiabatic",),
    "flux_dop853": ("flux",),
    "slow_envelope": ("slow",),
    "dyson": ("plateau",),
    "single_tone_rwa": ("plateau",),
    "floquet_period": ("plateau",),
    "t6": ("plateau",),
    "single_tone_gate": ("shaped",),
    "shaped_sambe": ("shaped",),
    "gate_cache": ("shaped",),
    "period_table": ("shaped",),
    "common_frame": ("two", "multiple"),
    "multitone_gate": ("multiple",),
    "multitone_rwa": ("two", "multiple"),
    "floquet": ("two",),
    "dop853": ("shaped", "two", "multiple"),
}


@dataclass(frozen=True)
class NumericWorkload:
    """Detached full input and provenance; structurally consumed by diagnostics.

    source owns reusable numeric physics, batch contains all requested points.
    name identifies the workload. metadata explains its representation; manifest
    records tracked inputs, calibration and compilation. domain_spans counts the
    public input domains; family_domains lists public candidate domains, and
    companions declares fixed extra family/domain ownership.
    """

    name: str
    source: ModelSource
    batch: TimelineBatch
    metadata: dict[str, object]
    manifest: dict[str, object]
    domain_spans: dict[str, int]
    family_domains: dict[str, tuple[str, ...]]
    companions: dict[str, tuple[str, ...]]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(_ROOT), *args], text=True).strip()


def _carrier_domains(batch: TimelineBatch) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for point, spans in enumerate(segment_spans(batch)):
        for start, end in spans:
            middle = (start + end) / 2
            active = [
                track
                for track in batch.tracks
                if track.active[point]
                and track.amplitude[point] != 0
                and track.start_ns[point]
                <= middle
                < track.start_ns[point] + track.duration_ns[point]
            ]
            if not active:
                counts["idle"] += 1
                continue
            if len(active) != 1:
                raise ValueError("carrier workload requires one tone per driven span")
            (track,) = active
            if track.envelopes[track.envelope_index[point]].kind != "constant":
                raise ValueError("carrier workload requires a constant probe envelope")
            counts["slow" if abs(track.frequency_ghz[point]) < 0.03 else "plateau"] += 1
    return dict(counts)


def carrier_workload(*, evidence_dir: Path | None = None) -> NumericWorkload:
    """Build the 24-carrier single-fluxonium workload using the real FreqExp.run.

    The 21-level model/profile come from HEAD Git blobs, not working copies.
    The carrier center is the DC model's E1-E0 in MHz. The +/-100 MHz span
    uses the documented 5 MHz initial linewidth guess and the form's 20x range.
    Gain 0.1 and constant 1 us length are the form's uncalibrated starting values.
    Readout frequency/flight time come from the tracked model; gain 0.1 and
    1 us integration are explicit starting values. No pi calibration is needed.
    Reps/rounds are one, pacing and drift off, seed retained from the template.

    evidence_dir, when supplied, retains replay JSON and live PNGs for review.
    Otherwise a temporary archive is consumed and removed before return.
    The returned source remains usable after backend cleanup and is caller-owned.
    Public compilation/acquisition/model/storage failures propagate unchanged.
    """
    if evidence_dir is None:
        # Replay requires a filesystem archive; automatic cleanup bounds its life.
        with TemporaryDirectory(prefix="family-pareto-carrier-") as directory:
            return _build_carrier(Path(directory))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return _build_carrier(evidence_dir)


def _build_carrier(directory: Path) -> NumericWorkload:
    settings = parse_single_fluxonium(yaml.safe_load(_git("show", f"HEAD:{_SETTINGS}")))
    settings = settings.model_copy(
        update={
            "drift": None,
            "numerics": settings.numerics.model_copy(update={"time_scale": 0.0}),
        }
    )
    board = build_soccfg(yaml.safe_load(_git("show", f"HEAD:{_PROFILE}")))
    env = single_fluxonium(settings, soccfg=board, private_replay=directory.resolve())
    try:
        with closing(SingleFluxoniumSource(settings, env.devices)) as initial_owner:
            initial = initial_owner.model_facts(initial_owner.snapshot()).source
            energies = initial.model((0,)).energies_ghz
            center_mhz = float((energies[1] - energies[0]) * 1000)
        (line,) = settings.drive_lines.values()
        cfg = FreqCfg(
            reset=None,
            init_pulse=None,
            qub_pulse=PulseCfg(
                ch=line.gen_ch,
                nqz=1,
                freq=center_mhz,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
            ),
            readout=PulseReadoutCfg(
                gen_ch=settings.readout.drive.gen_ch,
                ro_ch=settings.readout.adc.ro_ch,
                nqz=1,
                freq=settings.readout.resonator.frequency_ghz * 1000,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
                ro_length=1.0,
                trig_offset=settings.readout.adc.time_of_flight_ns / 1000,
            ),
            freq=SweepCfg(
                start=center_mhz - 100,
                stop=center_mhz + 100,
                step=200 / 23,
                expts=24,
            ),
            reps=1,
            rounds=1,
            relax_delay=1.0,
        )
        acquired = run_stage(FreqExp(), cfg, env)
        for name, png in acquired.pngs.items():
            (directory / name).write_bytes(png)
        (round_path,) = replay_paths(directory)
        replay = read_replay(round_path)
        if not isinstance(replay.acquire.settings, SingleFluxoniumSettings):
            raise ValueError("carrier replay has a different physical backend")
        with closing(
            SingleFluxoniumSource(replay.acquire.settings, DeviceManager())
        ) as owner:
            facts = owner.model_facts(replay.round.snapshot)
            request = replay.acquire.request
            (template,) = request.templates
            batch = owner.plan_template(
                replay.round.snapshot, template, time_base=request.time_base
            ).batch
            domains = _carrier_domains(batch)
            return NumericWorkload(
                "single_fluxonium_carrier",
                facts.source,
                batch,
                {
                    "reason": "Real twotone/freq carrier scan on the tracked 21-level model",
                    "experiment": "twotone/freq",
                    "sweep_type": "F",
                },
                {
                    "program_source": "experiment_run",
                    "blobs": {
                        path: _git("rev-parse", f"HEAD:{path}")
                        for path in (_SETTINGS, _PROFILE, _EXPERIMENT, _FORM)
                    },
                    "configuration": cfg.model_dump(mode="json"),
                    "effective_settings": settings.model_dump(mode="json"),
                    "snapshot": asdict(replay.round.snapshot),
                    "time_base": {
                        "ticks_per_us_numerator": request.time_base.ticks_per_us.numerator,
                        "ticks_per_us_denominator": request.time_base.ticks_per_us.denominator,
                    },
                    "points": len(batch.points),
                    "spans": sum(domains.values()),
                    "domain_spans": domains,
                    "waveform": "constant",
                    "pulse_length_us": 1.0,
                    "requested_carrier_range_mhz": [center_mhz - 100, center_mhz + 100],
                    "compiled_carriers_ghz": [
                        track.frequency_ghz.tolist() for track in batch.tracks
                    ],
                    "calibration": {
                        "center_mhz": center_mhz,
                        "center_method": "DC model E1-E0",
                        "linewidth_guess_mhz": 5.0,
                        "linewidth_method": "fixed documented initial guess, not measured",
                        "probe_gain": 0.1,
                        "probe_gain_method": "uncalibrated form starting value; no pi pulse used",
                        "readout_mhz": settings.readout.resonator.frequency_ghz * 1000,
                        "readout_method": "tracked bare cavity frequency, gain 0.1, 1 us pulse/window, tracked flight time",
                    },
                },
                domains,
                dict(_FAMILY_DOMAINS),
                {"t4": ("idle",)},
            )
    finally:
        try:
            if env.close_backend is not None:
                env.close_backend()
        finally:
            env.devices.close_all_devices()
