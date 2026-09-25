from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner import ScheduleOutcome
from zcu_tools.gui.app.autofluxdep.cfg import (
    OverridePlan,
    RunCfgSnapshot,
    apply_override_patches,
)
from zcu_tools.gui.app.autofluxdep.experiments._support import acquire as acquire_mod
from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.program.v2 import SweepCfg


def test_run_env_knob_reads_run_start_snapshot() -> None:
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        node_name="qubit_freq",
        knobs_snapshot={"acquire_retry": 2},
    )

    assert env.knob("acquire_retry") == 2
    assert env.knob("missing", "fallback") == "fallback"
    with pytest.raises(KeyError, match="missing"):
        env.knob("missing")

    mutable_copy = env.knobs()
    mutable_copy["acquire_retry"] = 99
    assert env.knob("acquire_retry") == 2


def test_run_cfg_snapshot_defensively_copies_nested_run_truth() -> None:
    samples = np.array([1.0, 2.0])
    base_cfg = {
        "modules": {"drive": {"gain": 0.1}},
        "channels": [0, 1],
        "bounds": (1.0, 2.0),
        "labels": {"a", "b"},
        "samples": samples,
    }
    trace = np.array([3.0, 4.0])
    knobs = {"acquire": {"retry": 2}, "trace": trace}
    snapshot = RunCfgSnapshot(
        base_cfg=base_cfg,
        override_plan=OverridePlan(),
        knobs=knobs,
    )

    base_cfg["modules"]["drive"]["gain"] = 0.9
    base_cfg["channels"].append(2)
    knobs["acquire"]["retry"] = 99
    samples[0] = 9.0
    trace[0] = 8.0

    snapshot_modules = cast(dict[str, Any], snapshot.base_cfg["modules"])
    snapshot_drive = cast(dict[str, Any], snapshot_modules["drive"])
    assert snapshot_drive["gain"] == 0.1
    assert snapshot.base_cfg["channels"] == (0, 1)
    assert snapshot.base_cfg["bounds"] == (1.0, 2.0)
    assert snapshot.base_cfg["labels"] == frozenset({"a", "b"})
    assert snapshot.knobs["acquire"]["retry"] == 2
    snapshot_samples = cast(np.ndarray[Any, Any], snapshot.base_cfg["samples"])
    snapshot_trace = cast(np.ndarray[Any, Any], snapshot.knobs["trace"])
    np.testing.assert_array_equal(snapshot_samples, [1.0, 2.0])
    np.testing.assert_array_equal(snapshot_trace, [3.0, 4.0])
    assert snapshot_samples.shape == samples.shape
    assert snapshot_samples.dtype == samples.dtype
    assert not snapshot_samples.flags.writeable
    assert not snapshot_trace.flags.writeable
    with pytest.raises(TypeError):
        snapshot_drive["gain"] = 0.5
    with pytest.raises(TypeError):
        snapshot.knobs["acquire"]["retry"] = 3
    with pytest.raises(ValueError, match="read-only"):
        snapshot_samples[0] = 5.0
    with pytest.raises(ValueError):
        snapshot_samples.setflags(write=True)

    wire_base = cast(dict[str, Any], snapshot.to_wire()["base_cfg"])
    assert isinstance(wire_base["channels"], list)
    assert isinstance(wire_base["bounds"], tuple)
    assert isinstance(wire_base["labels"], set)
    wire_samples = cast(np.ndarray[Any, Any], wire_base["samples"])
    assert wire_samples.flags.writeable
    wire_samples[0] = 6.0
    assert snapshot_samples[0] == 1.0

    first = apply_override_patches(
        snapshot.base_cfg,
        OverridePlan(),
        {},
        flux_idx=0,
        node_name="test",
    )
    first_modules = cast(dict[str, Any], first["modules"])
    first_drive = cast(dict[str, Any], first_modules["drive"])
    assert isinstance(first["channels"], list)
    assert isinstance(first["bounds"], tuple)
    assert isinstance(first["labels"], set)
    first_samples = cast(np.ndarray[Any, Any], first["samples"])
    assert first_samples.flags.writeable
    first_drive["gain"] = 0.7
    first_samples[0] = 7.0
    second = apply_override_patches(
        snapshot.base_cfg,
        OverridePlan(),
        {},
        flux_idx=1,
        node_name="test",
    )
    second_modules = cast(dict[str, Any], second["modules"])
    second_drive = cast(dict[str, Any], second_modules["drive"])
    second_samples = cast(np.ndarray[Any, Any], second["samples"])
    assert second_drive["gain"] == 0.1
    assert second_samples[0] == 1.0


def test_sweep_cfg_snapshot_is_frozen_and_thaws_to_real_sweep_cfg() -> None:
    source = SweepCfg(start=0.0, stop=2.0, expts=3, step=1.0)
    snapshot = RunCfgSnapshot(
        base_cfg={"sweep": source},
        override_plan=OverridePlan(),
        knobs={"sweep": source},
    )
    schema = QubitFreqBuilder().make_default_schema()
    first = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot=snapshot.knobs,
    )
    second = RunEnv(
        flux=0.1,
        flux_idx=1,
        schema=schema,
        knobs_snapshot=snapshot.knobs,
    )

    first_sweep = first.knob("sweep")
    second_sweep = second.knob("sweep")
    assert first_sweep is second_sweep
    assert (
        first_sweep.start,
        first_sweep.stop,
        first_sweep.expts,
        first_sweep.step,
    ) == (
        0.0,
        2.0,
        3,
        1.0,
    )
    with pytest.raises(FrozenInstanceError):
        first_sweep.stop = 3.0
    assert first_sweep.stop == second_sweep.stop == 2.0

    with pytest.raises(ValidationError):
        source.stop = 3.0
    assert first_sweep.stop == second_sweep.stop == 2.0

    mutable_sweep = first.knobs()["sweep"]
    assert isinstance(mutable_sweep, SweepCfg)
    assert mutable_sweep.stop == 2.0
    with pytest.raises(ValidationError):
        mutable_sweep.stop = 3.0
    assert first_sweep.stop == second_sweep.stop == 2.0

    wire_sweep = cast(dict[str, Any], snapshot.to_wire()["base_cfg"])["sweep"]
    point_sweep = apply_override_patches(
        snapshot.base_cfg,
        OverridePlan(),
        {},
        flux_idx=0,
        node_name="test",
    )["sweep"]
    assert isinstance(wire_sweep, SweepCfg)
    assert isinstance(point_sweep, SweepCfg)
    assert wire_sweep.stop == point_sweep.stop == 2.0


def test_snapshot_rejects_unsupported_mutable_leaf_types() -> None:
    class OtherModel(BaseModel):
        value: int

    with pytest.raises(
        TypeError, match="unsupported run snapshot leaf type.*OtherModel"
    ):
        RunCfgSnapshot(
            base_cfg={},
            override_plan=OverridePlan(),
            knobs={"other": OtherModel(value=1)},
        )
    with pytest.raises(TypeError, match="object dtype is unsupported"):
        RunCfgSnapshot(
            base_cfg={"objects": np.array([{"mutable": True}], dtype=object)},
            override_plan=OverridePlan(),
            knobs={},
        )


def test_run_env_reuses_internal_frozen_knobs_but_freezes_external_mappings() -> None:
    schema = QubitFreqBuilder().make_default_schema()
    snapshot = RunCfgSnapshot(
        base_cfg={},
        override_plan=OverridePlan(),
        knobs={"feedback": {"gain": 0.1}},
    )

    from_snapshot = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot=snapshot.knobs,
    )
    assert from_snapshot.knobs_view() is snapshot.knobs

    backing = {"feedback": {"gain": 0.2}}
    external = MappingProxyType(backing)
    from_external = RunEnv(
        flux=0.1,
        flux_idx=1,
        schema=schema,
        knobs_snapshot=external,
    )
    assert from_external.knobs_view() is not external
    backing["feedback"]["gain"] = 0.9
    assert from_external.knob("feedback")["gain"] == 0.2


def test_run_env_direct_knobs_snapshot_is_immutable_and_flux_independent() -> None:
    source = {"feedback": {"gain": 0.1}}
    schema = QubitFreqBuilder().make_default_schema()
    first = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot=source,
    )
    second = RunEnv(
        flux=0.1,
        flux_idx=1,
        schema=schema,
        knobs_snapshot=source,
    )

    source["feedback"]["gain"] = 0.9
    assert first.knob("feedback")["gain"] == 0.1
    assert second.knob("feedback")["gain"] == 0.1
    with pytest.raises(TypeError):
        first.knobs_view()["feedback"]["gain"] = 0.5

    mutable_copy = first.knobs()
    mutable_copy["feedback"]["gain"] = 0.7
    assert second.knob("feedback")["gain"] == 0.1


def test_snr_stop_condition_reads_cached_snr_at_original_cadence(
    monkeypatch: Any,
) -> None:
    now = {"value": 1.0}
    monkeypatch.setattr(acquire_mod.time, "time", lambda: now["value"])
    probe = acquire_mod.SnrProbe()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        knobs_snapshot={"earlystop_snr": 50.0},
    )

    stop_condition = acquire_mod.build_stop_condition(env, probe)

    assert stop_condition is not None
    assert stop_condition() is False

    probe.value = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    probe.snr = 49.9
    assert stop_condition() is False

    probe.snr = 50.0
    now["value"] = 1.05
    assert stop_condition() is False

    now["value"] = 1.101
    assert stop_condition() is True

    probe.snr = 0.0
    assert stop_condition() is True


def test_snr_stop_condition_uses_run_start_knob_snapshot() -> None:
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        knobs_snapshot={"earlystop_snr": None},
    )

    stop_condition = acquire_mod.build_stop_condition(env, acquire_mod.SnrProbe())

    assert stop_condition is None


def test_setup_flux_point_updates_selected_device_before_setup(
    monkeypatch: Any,
) -> None:
    cfg = ExpCfgModel(dev={"flux": FakeDeviceInfo(address="none")})
    env = RunEnv(
        flux=0.25,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        flux_device="flux",
    )
    setup_calls: list[tuple[ExpCfgModel, bool]] = []

    def record_setup(value: ExpCfgModel, *, progress: bool) -> None:
        setup_calls.append((value, progress))

    monkeypatch.setattr(acquire_mod, "setup_devices", record_setup)

    acquire_mod.setup_flux_point(cfg, env, "qubit_freq")

    assert cfg.dev is not None
    assert cast(FakeDeviceInfo, cfg.dev["flux"]).value == pytest.approx(0.25)
    assert setup_calls == [(cfg, False)]


def test_setup_flux_point_fast_fails_without_selected_device() -> None:
    cfg = ExpCfgModel(dev={"flux": FakeDeviceInfo(address="none")})
    env = RunEnv(
        flux=0.25,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        flux_device=None,
    )

    with pytest.raises(RuntimeError, match="needs a flux device picked"):
        acquire_mod.setup_flux_point(cfg, env, "qubit_freq")


def test_schedule_completed_distinguishes_completed_and_stopped() -> None:
    assert acquire_mod.schedule_completed(ScheduleOutcome(), "t1")
    assert not acquire_mod.schedule_completed(
        ScheduleOutcome(status="stopped", reason="operator stopped"), "t1"
    )


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_schedule_completed_preserves_failure_reason_and_cause(status: Any) -> None:
    cause = ValueError("hardware")
    outcome = ScheduleOutcome(status=status, reason="acquire aborted", exception=cause)

    with pytest.raises(RuntimeError, match="acquire aborted") as exc_info:
        acquire_mod.schedule_completed(outcome, "t1")

    assert exc_info.value.__cause__ is cause


def test_schedule_completed_rejects_unknown_status() -> None:
    outcome = ScheduleOutcome(status=cast(Any, "paused"))

    with pytest.raises(RuntimeError, match="unsupported t1 Schedule outcome"):
        acquire_mod.schedule_completed(outcome, "t1")


def test_acquire_retry_reads_default_and_validates_non_negative():
    schema = QubitFreqBuilder().make_default_schema()
    env = RunEnv(flux=0.0, flux_idx=0, schema=schema, knobs_snapshot={})
    assert acquire_mod.acquire_retry(env) == acquire_mod.DEFAULT_ACQUIRE_RETRY

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot={"acquire_retry": 0},
    )
    assert acquire_mod.acquire_retry(env) == 0

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot={"acquire_retry": 2},
    )
    assert acquire_mod.acquire_retry(env) == 2

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        node_name="qubit_freq",
        knobs_snapshot={"acquire_retry": -1},
    )
    with pytest.raises(RuntimeError, match="acquire_retry must be non-negative"):
        acquire_mod.acquire_retry(env)
