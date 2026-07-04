from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
from zcu_tools.gui.app.autofluxdep.feedback import (
    FeedbackRuntime,
    IdwEstimator,
    LastGoodEstimator,
    LogStepController,
    build_feedback_runtime,
)
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder


def test_last_good_estimator_returns_latest_observation_for_any_flux():
    estimator = LastGoodEstimator()

    assert estimator.estimate(0.0) is None

    estimator.observe(0.0, 1.5)
    estimator.observe(0.1, 2.5)

    assert estimator.estimate(-10.0) == pytest.approx(2.5)
    assert estimator.estimate(10.0) == pytest.approx(2.5)


def test_idw_estimator_returns_none_without_observations_and_interpolates():
    estimator = IdwEstimator(k=4, epsilon=1e-6)

    assert estimator.estimate(0.0) is None

    estimator.observe(0.0, 1.0)
    assert estimator.estimate(0.5) == pytest.approx(1.0)

    estimator.observe(1.0, 3.0)
    assert estimator.estimate(0.5) == pytest.approx(2.0)

    with pytest.raises(RuntimeError, match="idw_k"):
        IdwEstimator(k=1.5)  # type: ignore[arg-type]


def test_log_step_controller_proposes_in_log_domain_and_fast_fails_invalid_input():
    controller = LogStepController(step_gain=1.0)

    proposal = controller.propose(0.2, math.log(2.0))

    assert proposal == pytest.approx(0.4)
    assert controller.latest() == pytest.approx(0.4)

    with pytest.raises(RuntimeError, match="current actuator"):
        controller.propose(0.0, 0.1)
    with pytest.raises(RuntimeError, match="normalized_error"):
        controller.propose(0.2, float("nan"))
    with pytest.raises(RuntimeError, match="proposal"):
        controller.propose(0.2, 1000.0)
    with pytest.raises(RuntimeError, match="step_gain"):
        LogStepController(step_gain=0.0)


@dataclass
class _Provider:
    name: str
    builder: Any
    schema: Any


def test_disabled_slot_returns_none_and_undeclared_slot_fast_fails():
    builder = QubitFreqBuilder()
    schema = builder.make_default_schema()
    schema.set_field("pred_freq_correction_enabled", False)
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
    schema_b.set_field("pred_freq_correction_strategy", "last_good")
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

    assert estimator_a.estimate(1.0) == pytest.approx(1.0)
    assert estimator_b.estimate(1.0) == pytest.approx(20.0)


def test_feedback_policy_strategy_and_full_values_persist_flat():
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("pred_freq_correction_strategy", "last_good")
    schema.set_field("pred_freq_correction_idw_k", 7)
    schema.set_field("pred_freq_correction_idw_epsilon", 1e-4)

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

    restored = QubitFreqBuilder().make_default_schema()
    restored.restore_persisted_raw(raw)

    knobs = restored.lower(None)
    assert knobs["pred_freq_correction_strategy"] == "last_good"
    assert knobs["pred_freq_correction_idw_k"] == 7
    assert knobs["pred_freq_correction_idw_epsilon"] == pytest.approx(1e-4)

    restored.set_field("pred_freq_correction_strategy", "not_a_strategy")
    with pytest.raises(RuntimeError, match="allowed choices"):
        restored.lower(None)
