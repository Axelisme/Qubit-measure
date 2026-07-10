from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
from zcu_tools.gui.app.autofluxdep.experiments.lenrabi import LenRabiBuilder
from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.feedback import (
    FeedbackRuntime,
    FeedbackSample,
    IdwEstimator,
    LastGoodEstimator,
    LogStepController,
    build_feedback_runtime,
)


def test_last_good_estimator_returns_latest_observation_for_any_flux():
    estimator = LastGoodEstimator(decay_points=3.0)

    assert estimator.estimate(0.0) is None

    estimator.observe(0.0, 1.5)
    estimator.observe(0.1, 2.5)

    first = estimator.estimate(-10.0)
    second = estimator.estimate(10.0)

    assert first == FeedbackSample(value=2.5, confidence=1.0, age_queries=0)
    assert second is not None
    assert second.value == pytest.approx(2.5)
    assert second.confidence == pytest.approx(math.exp(-1.0 / 3.0))
    assert second.age_queries == 1


def test_idw_estimator_returns_none_without_observations_and_interpolates():
    estimator = IdwEstimator(k=4, epsilon=1e-6, decay_points=4.0)

    assert estimator.estimate(0.0) is None

    estimator.observe(0.0, 1.0)
    first = estimator.estimate(0.5)
    assert first is not None
    assert first.value == pytest.approx(1.0)
    assert first.confidence == pytest.approx(1.0)

    estimator.observe(1.0, 3.0)
    second = estimator.estimate(0.5)
    assert second is not None
    assert second.value == pytest.approx(2.0)
    assert second.confidence == pytest.approx(1.0)

    stale = estimator.estimate(0.5)
    assert stale is not None
    assert stale.value == pytest.approx(2.0)
    assert stale.confidence == pytest.approx(math.exp(-1.0 / 4.0))
    assert stale.age_queries == 1

    with pytest.raises(RuntimeError, match="idw_k"):
        IdwEstimator(k=1.5)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="decay_points"):
        IdwEstimator(decay_points=0.0)


def test_last_good_replace_observations_reseeds_value_and_age():
    estimator = LastGoodEstimator(decay_points=3.0)
    estimator.observe(0.0, 1.0)
    assert estimator.estimate(0.0) is not None
    assert estimator.estimate(0.0) is not None

    estimator.replace_observations(((0.1, 5.0), (0.2, 7.0)))

    first = estimator.estimate(10.0)
    assert first == FeedbackSample(value=7.0, confidence=1.0, age_queries=0)
    second = estimator.estimate(10.0)
    assert second is not None
    assert second.confidence == pytest.approx(math.exp(-1.0 / 3.0))

    estimator.replace_observations(())
    assert estimator.estimate(0.0) is None


def test_idw_replace_observations_rebuilds_points_and_age():
    estimator = IdwEstimator(k=4, epsilon=1e-6, decay_points=4.0)
    estimator.observe(0.0, 100.0)
    assert estimator.estimate(0.0) is not None
    assert estimator.estimate(0.0) is not None

    estimator.replace_observations(((0.0, 1.0), (1.0, 3.0)))

    first = estimator.estimate(0.5)
    assert first is not None
    assert first.value == pytest.approx(2.0)
    assert first.confidence == pytest.approx(1.0)
    second = estimator.estimate(0.5)
    assert second is not None
    assert second.confidence == pytest.approx(math.exp(-1.0 / 4.0))

    estimator.replace_observations(())
    assert estimator.estimate(0.5) is None


def test_log_step_controller_proposes_in_log_domain_and_fast_fails_invalid_input():
    controller = LogStepController(step_gain=1.0)

    proposal = controller.propose(0.2, math.log(2.0))

    assert proposal.value == pytest.approx(0.4)
    assert proposal.confidence == pytest.approx(1.0)
    latest = controller.latest()
    assert latest is not None
    assert latest.value == pytest.approx(0.4)
    assert latest.confidence == pytest.approx(1.0)

    stale = controller.latest()
    assert stale is not None
    assert stale.value == pytest.approx(0.4)
    assert stale.confidence == pytest.approx(math.exp(-1.0 / controller.decay_points))

    with pytest.raises(RuntimeError, match="current actuator"):
        controller.propose(0.0, 0.1)
    with pytest.raises(RuntimeError, match="normalized_error"):
        controller.propose(0.2, float("nan"))
    with pytest.raises(RuntimeError, match="proposal"):
        controller.propose(0.2, 1000.0)
    with pytest.raises(RuntimeError, match="step_gain"):
        LogStepController(step_gain=0.0)
    with pytest.raises(RuntimeError, match="decay_points"):
        LogStepController(decay_points=0.0)


@dataclass
class _Provider:
    name: str
    builder: Any
    schema: Any


@dataclass
class _RawSchema:
    knobs: dict[str, Any]

    def lower(self, ml: Any, md: Any = None) -> dict[str, Any]:
        del ml, md
        return dict(self.knobs)


def test_disabled_slot_returns_none_and_undeclared_slot_fast_fails():
    builder = QubitFreqBuilder()
    schema = builder.make_default_schema()
    schema.set_field("pred_freq_correction_strategy", "off")
    runtime = build_feedback_runtime([_Provider("qf", builder, schema)])

    view = runtime.view_for("qf")

    assert view.estimator("predict_freq_correction") is None
    with pytest.raises(KeyError, match="undeclared feedback slot"):
        view.estimator("missing")
    with pytest.raises(KeyError, match="undeclared feedback slot"):
        FeedbackRuntime().view_for("qf").controller("predict_freq_correction")


def test_same_builder_type_placements_get_independent_feedback_state():
    builder = QubitFreqBuilder()
    schema_a = builder.make_default_schema()
    schema_b = builder.make_default_schema()
    schema_a.set_field("pred_freq_correction_decay_points", 2.0)
    schema_b.set_field("pred_freq_correction_strategy", "last_good")
    schema_b.set_field("pred_freq_correction_decay_points", 8.0)
    runtime = build_feedback_runtime(
        [_Provider("qf_a", builder, schema_a), _Provider("qf_b", builder, schema_b)]
    )

    estimator_a = runtime.view_for("qf_a").estimator("predict_freq_correction")
    estimator_b = runtime.view_for("qf_b").estimator("predict_freq_correction")
    assert estimator_a is not None
    assert estimator_b is not None
    assert estimator_a is not estimator_b

    estimator_a.observe(0.0, 1.0)
    estimator_b.observe(0.0, 10.0)
    estimator_b.observe(1.0, 20.0)

    estimate_a = estimator_a.estimate(1.0)
    estimate_b = estimator_b.estimate(1.0)
    assert estimate_a is not None
    assert estimate_b is not None
    assert estimate_a.value == pytest.approx(1.0)
    assert estimate_b.value == pytest.approx(20.0)

    stale_a = estimator_a.estimate(1.0)
    stale_b = estimator_b.estimate(1.0)
    assert stale_a is not None
    assert stale_b is not None
    assert stale_a.confidence == pytest.approx(math.exp(-1.0 / 2.0))
    assert stale_b.confidence == pytest.approx(math.exp(-1.0 / 8.0))


def test_feedback_policy_strategy_and_full_values_persist_flat():
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("pred_freq_correction_strategy", "last_good")
    schema.set_field("pred_freq_correction_idw_k", 7)
    schema.set_field("pred_freq_correction_idw_epsilon", 1e-4)
    schema.set_field("pred_freq_correction_decay_points", 2.5)

    raw = schema.to_persisted_raw()

    generation = raw["generation"]
    assert isinstance(generation, dict)
    assert "feedback" not in generation
    assert generation["pred_freq_correction_strategy"] == {
        "__kind": "direct",
        "value": "last_good",
    }
    assert generation["pred_freq_correction_idw_k"] == {
        "__kind": "direct",
        "value": 7,
    }
    assert generation["pred_freq_correction_idw_epsilon"] == {
        "__kind": "direct",
        "value": 1e-4,
    }
    assert generation["pred_freq_correction_decay_points"] == {
        "__kind": "direct",
        "value": 2.5,
    }

    restored = QubitFreqBuilder().make_default_schema()
    restored.restore_persisted_raw(raw)

    knobs = restored.lower(None)
    assert knobs["pred_freq_correction_strategy"] == "last_good"
    assert knobs["pred_freq_correction_idw_k"] == 7
    assert knobs["pred_freq_correction_idw_epsilon"] == pytest.approx(1e-4)
    assert knobs["pred_freq_correction_decay_points"] == pytest.approx(2.5)

    restored.set_field("pred_freq_correction_strategy", "not_a_strategy")
    with pytest.raises(RuntimeError, match="allowed choices"):
        restored.lower(None)


def test_controller_feedback_policy_values_persist_flat():
    schema = LenRabiBuilder().make_default_schema()
    schema.set_field("pi_gain_feedback_strategy", "off")

    raw = schema.to_persisted_raw()

    generation = raw["generation"]
    assert isinstance(generation, dict)
    assert "feedback" not in generation
    assert generation["pi_gain_feedback_strategy"] == {
        "__kind": "direct",
        "value": "off",
    }
    assert "pi_gain_feedback_step_gain" not in generation
    assert "pi_gain_feedback_decay_points" not in generation

    restored = LenRabiBuilder().make_default_schema()
    restored.restore_persisted_raw(raw)

    knobs = restored.lower(None)
    assert knobs["pi_gain_feedback_strategy"] == "off"
    assert "pi_gain_feedback_step_gain" not in knobs
    assert "pi_gain_feedback_decay_points" not in knobs

    runtime = build_feedback_runtime([_Provider("lenrabi", LenRabiBuilder(), restored)])
    assert runtime.view_for("lenrabi").controller("drive_gain") is None


def test_controller_feedback_uses_declared_defaults_for_hidden_tuning():
    schema = LenRabiBuilder().make_default_schema()

    runtime = build_feedback_runtime([_Provider("lenrabi", LenRabiBuilder(), schema)])
    controller = runtime.view_for("lenrabi").controller("drive_gain")

    assert isinstance(controller, LogStepController)
    assert controller.step_gain == pytest.approx(0.5)
    assert controller.decay_points == pytest.approx(3.0)


def test_controller_feedback_ignores_removed_legacy_tuning_keys():
    builder = LenRabiBuilder()
    slot = builder.feedback_slots[0]
    schema = _RawSchema(
        {
            "pi_gain_feedback_strategy": "log_step",
            "pi_gain_feedback_step_gain": 99.0,
            "pi_gain_feedback_decay_points": 99.0,
        }
    )

    runtime = build_feedback_runtime([_Provider("lenrabi", builder, schema)])
    controller = runtime.view_for("lenrabi").controller("drive_gain")

    assert isinstance(controller, LogStepController)
    assert controller.step_gain == pytest.approx(slot.default_step_gain)
    assert controller.decay_points == pytest.approx(slot.default_decay_points)
