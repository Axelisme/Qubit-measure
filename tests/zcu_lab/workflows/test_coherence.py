"""Coherence behavior through the public Engine and concrete native readers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError
from zcu_tools.datafile import JsonObject, SoftwareProvenance, load_run_data
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import Engine, EnginePorts, RunIdentity, RunPaths
from zcu_tools.plotting.plots import NonPresentingHost, Plots

import zcu_lab.workflows as workflows
from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiExp
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainExp
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyExp
from zcu_lab.workflows.calibration import Calibrated

from ._sim_inputs import Inputs, make_plan, seed_from_inputs


class OrderedDevices(RecordingDevices):
    """Number actual acquisition captures to observe the experiment order."""

    def capture_snapshot(self) -> dict[str, JsonObject]:
        super().capture_snapshot()
        return {
            "flux": {
                "value": self.values[-1][1],
                "value_unit": "native",
                "capture_seq": self.captures,
            }
        }


@contextmanager
def scan(
    inputs: Inputs,
    root: Path,
    plan: workflows.CoherencePlan,
    tun: workflows.CoherenceTunables,
) -> Iterator[Engine[Calibrated]]:
    """Own one real simulator Engine and release plots after the producer stops."""
    plots = Plots(NonPresentingHost())
    engine = Engine(
        EnginePorts(
            plots,
            RecordingBar,
            ManualClock(),
            context=seed_from_inputs(inputs),
            soc=inputs.soc,
            soccfg=inputs.soccfg,
            devices=OrderedDevices(),
        )
    )
    try:
        engine.start(
            workflows.coherence_fluxdep,
            plan=plan,
            tunables=tun,
            paths=RunPaths(root / "metadata", root / "data"),
            identity=RunIdentity(
                "coherence",
                SoftwareProvenance(
                    software_versions={},
                    git_commit=None,
                    git_dirty=None,
                    qick_version=None,
                    soc_fingerprint=None,
                    hostname="offline",
                ),
            ),
        )
        status = engine.execute("coherence")
        assert status.lifecycle == "done", status.reason
        yield engine
    finally:
        plots.finish()
        plots.release()


def plan_for(inputs: Inputs, **flags: bool) -> workflows.CoherencePlan:
    """Use the native module inputs at two flux points, with resolved plan flags."""
    return workflows.CoherencePlan(
        **make_plan(inputs).model_dump(mode="python"), **flags
    )


def knobs() -> workflows.CoherenceTunables:
    """Reduced averaging, a fine Ramsey grid, and deliberately rejected readout."""
    return workflows.CoherenceTunables.model_validate(
        {
            "cal": {
                "span_mhz": 2.0,
                "rabi_start_factor": 0.125,
                "min_readout_snr": 1e12,
                "ro_every": 2,
            },
            "t1": {"rounds": 2, "guess_us": 50.0},
            "t2ramsey": {"rounds": 10, "guess_us": 10.0, "detune_ratio": 0.25},
            "t2echo": {"rounds": 2, "guess_us": 15.0},
        }
    )


def test_three_enabled_lifetimes_keep_order_native_values_and_independent_timing(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = plan_for(inputs).model_copy(update={"points": 101})
    tun = knobs()
    with scan(inputs, tmp_path, plan, tun) as engine:
        records = engine.records("coherence")
        assert len(records) == 2
        accepted_t1 = None
        accepted_ramsey = None
        accepted_echo = None
        for record in records:
            point = record.encoded_record.value
            assert isinstance(point, dict)
            measurements = point["measurements"]
            assert isinstance(measurements, list)
            assert [m["kind"] for m in measurements if isinstance(m, dict)] == [
                "t1",
                "t2ramsey",
                "t2echo",
            ]
            for measurement, lifetime in zip(
                measurements, (20.0, 10.0, 10.0), strict=True
            ):
                assert isinstance(measurement, dict)
                assert measurement["status"] == "accepted", measurement
                assert measurement["lifetime_us"] == pytest.approx(lifetime, rel=0.3)
                assert measurement["error_us"] is not None
                assert measurement["file"] == f"{measurement['kind']}.h5"
            root = tmp_path / "data" / record.iteration_dir
            rabi, _ = load_run_with_metadata(
                root / "rabi.h5", spec=LenRabiExp().persistence_spec()
            )
            assert rabi.cfg is not None
            assert rabi.cfg.relax_delay == pytest.approx(
                250.0 if accepted_t1 is None else 5 * accepted_t1
            )
            t1, t1_meta = load_run_with_metadata(
                root / "t1.h5", spec=T1Exp().persistence_spec()
            )
            ramsey, ramsey_meta = load_run_with_metadata(
                root / "t2ramsey.h5", spec=T2RamseyExp().persistence_spec()
            )
            echo, echo_meta = load_run_with_metadata(
                root / "t2echo.h5", spec=T2EchoExp().persistence_spec()
            )
            assert (
                t1.cfg is not None and ramsey.cfg is not None and echo.cfg is not None
            )
            assert t1.cfg.sweep.length.stop == pytest.approx(
                250.0 if accepted_t1 is None else 5 * (25.0 + 0.5 * accepted_t1)
            )
            assert ramsey.cfg.sweep.length.stop == pytest.approx(
                50.0 if accepted_ramsey is None else 5 * (5.0 + 0.5 * accepted_ramsey)
            )
            assert echo.cfg.sweep.length.stop == pytest.approx(
                75.0 if accepted_echo is None else 5 * (7.5 + 0.5 * accepted_echo)
            )
            assert ramsey.cfg.relax_delay == 30.0 and echo.cfg.relax_delay == 45.0
            assert ramsey.cfg.detune == pytest.approx(
                0.25 / ramsey.cfg.sweep.length.step
            )
            assert (
                ramsey.result.true_activate_detune != 0
                and echo.result.true_activate_detune == 0
            )
            assert echo.cfg.detune == 0
            rabi_result = point["rabi"]
            assert isinstance(rabi_result, dict)
            assert ramsey.cfg.modules.pi2_pulse.get_param("length") == pytest.approx(
                rabi_result["pi2_length_us"]
            )
            assert echo.cfg.modules.pi_pulse.get_param("length") == pytest.approx(
                rabi_result["pi_length_us"]
            )
            assert echo.cfg.modules.pi2_pulse.get_param("length") == pytest.approx(
                rabi_result["pi2_length_us"]
            )
            files = (
                ["frequency-01.h5", "rabi.h5"]
                + (["readout.h5"] if point["index"] == 0 else [])
                + ["t1.h5", "t2ramsey.h5", "t2echo.h5"]
            )
            seq = []
            for filename in files:
                stored = load_run_data(root / filename)
                assert stored.metadata.snapshot == plan.snapshot
                assert stored.metadata.device_snapshot is not None
                seq.append(stored.metadata.device_snapshot["flux"]["capture_seq"])
            assert seq == sorted(seq)
            assert (
                t1_meta.device_snapshot
                != ramsey_meta.device_snapshot
                != echo_meta.device_snapshot
            )
            accepted_t1 = measurements[0]["lifetime_us"]
            accepted_ramsey = measurements[1]["lifetime_us"]
            accepted_echo = measurements[2]["lifetime_us"]


@pytest.mark.parametrize("enabled", ["t1", "t2ramsey", "t2echo"])
def test_flags_acquire_and_record_only_enabled_lifetime(
    inputs: Inputs, tmp_path: Path, enabled: str
) -> None:
    plan = plan_for(
        inputs,
        measure_t1=enabled == "t1",
        measure_t2ramsey=enabled == "t2ramsey",
        measure_t2echo=enabled == "t2echo",
    ).model_copy(update={"points": 101})
    with scan(inputs, tmp_path, plan, knobs()) as engine:
        for record in engine.records("coherence"):
            point = record.encoded_record.value
            assert isinstance(point, dict)
            measurements = point["measurements"]
            assert isinstance(measurements, list) and len(measurements) == 1
            member = measurements[0]
            assert isinstance(member, dict) and member["kind"] == enabled
            assert member["status"] == "accepted"
            root = tmp_path / "data" / record.iteration_dir
            lifetime_tags = [
                load_run_data(path).metadata.experiment
                for path in root.glob("*.h5")
                if load_run_data(path).metadata.experiment.startswith("twotone/ge/t")
            ]
            assert lifetime_tags == [f"twotone/ge/{enabled}"]
            rabi, _ = load_run_with_metadata(
                root / "rabi.h5", spec=LenRabiExp().persistence_spec()
            )
            assert rabi.cfg is not None
            if enabled != "t1":
                assert rabi.cfg.relax_delay == 250.0
            if point["index"] == 0:
                readout, _ = load_run_with_metadata(
                    root / "readout.h5", spec=FreqGainExp().persistence_spec()
                )
                assert readout.cfg is not None and readout.cfg.relax_delay == 250.0


def test_all_disabled_plan_is_invalid_before_acquisition(inputs: Inputs) -> None:
    plan_type = workflows.CoherencePlan
    with pytest.raises(ValidationError, match="At least one lifetime"):
        plan_type(
            **make_plan(inputs).model_dump(mode="python"),
            measure_t1=False,
            measure_t2ramsey=False,
            measure_t2echo=False,
        )


def test_rejected_t1_keeps_native_and_does_not_change_t2_or_shared_estimate(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = plan_for(inputs).model_copy(update={"points": 101})
    tun = knobs()
    tun.t1.max_rel_err = 1e-8
    with scan(inputs, tmp_path, plan, tun) as engine:
        for record in engine.records("coherence"):
            point = record.encoded_record.value
            assert isinstance(point, dict)
            members = point["measurements"]
            assert isinstance(members, list)
            first = members[0]
            assert isinstance(first, dict)
            assert first["status"] == "rejected" and first["file"] == "t1.h5"
            assert first["lifetime_us"] is None
            assert [m["status"] for m in members[1:] if isinstance(m, dict)] == [
                "accepted",
                "accepted",
            ]
            root = tmp_path / "data" / record.iteration_dir
            rabi, _ = load_run_with_metadata(
                root / "rabi.h5", spec=LenRabiExp().persistence_spec()
            )
            t1, _ = load_run_with_metadata(
                root / "t1.h5", spec=T1Exp().persistence_spec()
            )
            assert rabi.cfg is not None and rabi.cfg.relax_delay == 250.0
            assert t1.cfg is not None and t1.cfg.sweep.length.stop == 250.0


def test_rabi_failure_keeps_frequency_and_marks_enabled_dependents_skipped(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = plan_for(inputs)
    tun = knobs()
    tun.cal.rabi_start_factor = 0.001
    with scan(inputs, tmp_path, plan, tun) as engine:
        for record in engine.records("coherence"):
            point = record.encoded_record.value
            assert isinstance(point, dict)
            frequency = point["frequency"]
            assert isinstance(frequency, dict) and frequency["status"] == "accepted"
            assert frequency["frequency_mhz"] == pytest.approx(
                seed_from_inputs(inputs).f01_mhz, abs=10.0
            )
            assert frequency["files"] == ["frequency-01.h5"]
            readout = point["readout"]
            assert (
                isinstance(readout, dict)
                and readout["status"] == "skipped"
                and readout["file"] is None
            )
            members = point["measurements"]
            assert isinstance(members, list) and len(members) == 3
            for member in members:
                assert isinstance(member, dict)
                assert member["status"] == "skipped" and member["file"] is None
                assert member["lifetime_us"] is None and member["reason"]


def test_rejected_frequency_keeps_all_completed_attempts_and_no_prior_dependency(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = plan_for(inputs)
    tun = knobs()
    tun.cal.max_frequency_rel_err = 1e-8
    with scan(inputs, tmp_path, plan, tun) as engine:
        for record in engine.records("coherence"):
            point = record.encoded_record.value
            assert isinstance(point, dict)
            frequency = point["frequency"]
            assert isinstance(frequency, dict) and frequency["status"] == "rejected"
            assert frequency["frequency_mhz"] is None
            assert frequency["files"] == [
                "frequency-01.h5",
                "frequency-02.h5",
                "frequency-03.h5",
            ]
            rabi = point["rabi"]
            assert (
                isinstance(rabi, dict)
                and rabi["status"] == "skipped"
                and rabi["file"] is None
            )
            for filename in frequency["files"]:
                assert isinstance(filename, str)
                saved = load_run_data(
                    tmp_path / "data" / record.iteration_dir / filename
                )
                assert saved.metadata.snapshot == plan.snapshot
