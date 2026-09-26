"""Live-context and resolved-only finished-cfg lowering contracts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    lower_finished_cfg,
    lower_resolved_cfg,
)
from zcu_tools.meta_tool import MetaDict
from zcu_tools.program.v2 import SweepCfg


def _schema(
    spec_fields: dict[str, object], value_fields: dict[str, object]
) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields=spec_fields),  # type: ignore[arg-type]
        value=CfgSectionValue(fields=value_fields),  # type: ignore[arg-type]
    )


def _library(
    *,
    modules: dict[str, object] | None = None,
    waveforms: dict[str, object] | None = None,
) -> MagicMock:
    ml = MagicMock()
    ml.modules = {} if modules is None else modules
    ml.waveforms = {} if waveforms is None else waveforms
    return ml


def test_resolved_lowering_uses_per_node_cached_shapes_and_keeps_sources() -> None:
    from copy import deepcopy

    first = CfgSectionSpec(label="First", fields={"gain": ScalarSpec("Gain", float)})
    second = CfgSectionSpec(label="Second", fields={"count": ScalarSpec("Count", int)})
    schema = _schema(
        {
            "first": ReferenceSpec("module", [first, second]),
            "second": ReferenceSpec("module", [first, second]),
            "center": ScalarSpec("Center", complex),
            "optional": ScalarSpec("Optional", float, optional=True),
        },
        {
            "first": ReferenceValue(
                "same_library_key",
                CfgSectionValue({"gain": EvalValue("gain", resolved=0.25)}),
                resolved_label="First",
            ),
            "second": ReferenceValue(
                "same_library_key",
                CfgSectionValue({"count": DirectValue(3)}),
                resolved_label="Second",
            ),
            "center": EvalValue("center", resolved=1 + 2j),
            "optional": DirectValue(None),
        },
    )
    previous = deepcopy(schema)
    result = lower_resolved_cfg(
        schema, make_range=lambda start, stop, *, expts: (start, stop, expts)
    )
    assert result == {"first": {"gain": 0.25}, "second": {"count": 3}, "center": 1 + 2j}
    assert schema == previous


@pytest.mark.parametrize(
    "value",
    [
        EvalValue("source"),
        EvalValue("source", resolved=2.0, error="lookup failed"),
        EvalValue("source", resolved=2.0, validation_error="choice missing"),
        DirectValue(None, raw="1e", error="invalid input"),
        DirectValue(2.0, validation_error="choice missing"),
    ],
)
def test_resolved_lowering_rejects_unresolved_or_invalid_scalar(value) -> None:
    schema = _schema(
        {"frequency": ScalarSpec("Frequency", float)}, {"frequency": value}
    )
    with pytest.raises(RuntimeError, match="frequency"):
        lower_resolved_cfg(
            schema, make_range=lambda start, stop, *, expts: (start, stop, expts)
        )


@pytest.mark.parametrize(
    ("label", "error"),
    [(None, None), ("Missing", None), ("Shape", "catalog entry missing")],
)
def test_resolved_lowering_requires_valid_cached_reference_shape(label, error) -> None:
    shape = CfgSectionSpec(label="Shape", fields={"gain": ScalarSpec("Gain", float)})
    schema = _schema(
        {"drive": ReferenceSpec("module", [shape])},
        {
            "drive": ReferenceValue(
                "<Custom:Shape>",
                CfgSectionValue({"gain": DirectValue(0.25)}),
                resolved_label=label,
                error=error,
            )
        },
    )
    with pytest.raises(RuntimeError):
        lower_resolved_cfg(
            schema, make_range=lambda start, stop, *, expts: (start, stop, expts)
        )


@pytest.mark.parametrize(
    ("spec", "value", "message"),
    [
        (ScalarSpec("Count", int), EvalValue("count", resolved=2.5), "not compatible"),
        (
            ScalarSpec("Count", int, choices=[1, 2]),
            EvalValue("count", resolved=3),
            "allowed choices",
        ),
        (
            ScalarSpec("Optional", float, optional=True),
            EvalValue("optional"),
            "unresolved",
        ),
    ],
)
def test_resolved_lowering_keeps_static_type_and_choice_validation(
    spec, value, message
) -> None:
    schema = _schema({"value": spec}, {"value": value})
    with pytest.raises(RuntimeError, match=message):
        lower_resolved_cfg(
            schema, make_range=lambda start, stop, *, expts: (start, stop, expts)
        )


def test_resolved_lowering_ranges_keep_cached_edges_and_optional_ref() -> None:
    from copy import deepcopy

    schema = _schema(
        {
            "sweep": SweepSpec(),
            "centered": CenteredSweepSpec(),
            "disabled": ReferenceSpec(
                "module", [CfgSectionSpec(label="Shape")], optional=True
            ),
        },
        {
            "sweep": SweepValue(
                EvalValue("start", resolved=0.0),
                EvalValue("stop", resolved=1.0),
                4,
                DirectValue(1 / 3, raw="0.3"),
                auto_norm=False,
            ),
            "centered": CenteredSweepValue(EvalValue("center", resolved=5.0), 4.0, 3),
            "disabled": None,
        },
    )
    previous = deepcopy(schema)
    result = lower_resolved_cfg(
        schema, make_range=lambda start, stop, *, expts: (start, stop, expts)
    )
    assert result == {"sweep": (0.0, 1.0, 4), "centered": (3.0, 7.0, 3)}
    assert schema == previous


def test_resolved_lowering_rejects_incomplete_step_and_locked_center_mismatch() -> None:
    incomplete = _schema(
        {"sweep": SweepSpec()},
        {
            "sweep": SweepValue(
                0.0,
                1.0,
                4,
                DirectValue(None, raw="1e", error="invalid step"),
                auto_norm=False,
            )
        },
    )
    locked = _schema(
        {"sweep": CenteredSweepSpec(locked_center=0.0)},
        {"sweep": CenteredSweepValue(EvalValue("center", resolved=2.0), 4.0, 3)},
    )
    for schema, message in ((incomplete, "step"), (locked, "locked")):
        with pytest.raises(RuntimeError, match=message):
            lower_resolved_cfg(
                schema, make_range=lambda start, stop, *, expts: (start, stop, expts)
            )


def test_resolved_lowering_detaches_mutable_literal_results() -> None:
    literal = {"values": [1, 2]}
    schema = _schema({"fixed": LiteralSpec(literal)}, {"fixed": DirectValue(literal)})
    result = lower_resolved_cfg(
        schema, make_range=lambda start, stop, *, expts: (start, stop, expts)
    )
    literal["values"].append(3)
    assert result == {"fixed": {"values": [1, 2]}}
    fixed = result["fixed"]
    assert isinstance(fixed, dict)
    fixed["values"].append(4)
    assert literal == {"values": [1, 2, 3]}


def test_lowers_scalar_literal_optional_section_and_device() -> None:
    schema = _schema(
        {
            "literal": LiteralSpec("fixed"),
            "count": ScalarSpec("Count", int),
            "optional": ScalarSpec("Optional", float, optional=True),
            "section": CfgSectionSpec(fields={"enabled": ScalarSpec("Enabled", bool)}),
            "device": ScalarSpec(
                "Device", str, choices_source="devices", required=True
            ),
        },
        {
            "literal": DirectValue("fixed"),
            "count": DirectValue(3),
            "optional": DirectValue(None),
            "section": CfgSectionValue(fields={"enabled": DirectValue(True)}),
            "device": DirectValue("flux_yoko"),
        },
    )

    assert schema_to_raw_dict(schema, None, None) == {
        "literal": "fixed",
        "count": 3,
        "section": {"enabled": True},
        "device": "flux_yoko",
    }


def test_eval_snapshot_precedes_current_context_and_warns_on_drift(caplog) -> None:
    schema = _schema(
        {"frequency": ScalarSpec("Frequency", float)},
        {"frequency": EvalValue("q_f", resolved=5000.0)},
    )
    md = MetaDict()
    md.q_f = 6000.0

    with caplog.at_level("WARNING"):
        raw = schema_to_raw_dict(schema, md, None)

    assert raw == {"frequency": 5000.0}
    assert [record.message for record in caplog.records] == [
        "Config field 'frequency' (Frequency): EvalValue 'q_f' snapshot 5000.0 "
        "differs from current md evaluation 6000.0; using snapshot"
    ]


def test_unresolved_eval_without_context_uses_exact_error() -> None:
    schema = _schema(
        {"frequency": ScalarSpec("Frequency", float)},
        {"frequency": EvalValue("q_f")},
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, None)

    assert str(exc_info.value) == (
        "Config field 'frequency' (Frequency) expression 'q_f' is unresolved"
    )


def test_lowers_sweep_and_centered_sweep_to_sweep_cfg() -> None:
    schema = _schema(
        {
            "linear": SweepSpec("Linear"),
            "centered": CenteredSweepSpec("Centered"),
        },
        {
            "linear": SweepValue(1.0, 2.0, 5),
            "centered": CenteredSweepValue(center=10.0, span=4.0, expts=5),
        },
    )

    raw = schema_to_raw_dict(schema, None, None)

    linear = raw["linear"]
    centered = raw["centered"]
    assert isinstance(linear, SweepCfg)
    assert linear.model_dump() == {"start": 1.0, "stop": 2.0, "expts": 5, "step": 0.25}
    assert isinstance(centered, SweepCfg)
    assert centered.model_dump() == {
        "start": 8.0,
        "stop": 12.0,
        "expts": 5,
        "step": 1.0,
    }


def test_custom_reference_flattens_embedded_snapshot() -> None:
    pulse = CfgSectionSpec(
        label="Pulse",
        fields={
            "type": LiteralSpec("pulse"),
            "gain": ScalarSpec("Gain", float),
        },
    )
    schema = _schema(
        {"module": ReferenceSpec(kind="module", allowed=[pulse])},
        {
            "module": ReferenceValue(
                "<Custom:Pulse>",
                CfgSectionValue(
                    fields={
                        "type": DirectValue("pulse"),
                        "gain": DirectValue(0.25),
                    }
                ),
            )
        },
    )

    assert schema_to_raw_dict(schema, None, None) == {
        "module": {"type": "pulse", "gain": 0.25}
    }


def test_disabled_optional_reference_is_omitted() -> None:
    pulse = CfgSectionSpec(label="Pulse", fields={})
    schema = _schema(
        {"module": ReferenceSpec(kind="module", allowed=[pulse], optional=True)},
        {"module": None},
    )

    assert schema_to_raw_dict(schema, None, None) == {}


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        (
            ReferenceSpec(
                kind="module", allowed=[CfgSectionSpec(label="Pulse", fields={})]
            ),
            ReferenceValue("missing", CfgSectionValue()),
            "Unknown module reference: 'missing'",
        ),
        (
            ReferenceSpec(
                kind="waveform", allowed=[CfgSectionSpec(label="Const", fields={})]
            ),
            ReferenceValue("missing", CfgSectionValue()),
            "Unknown waveform reference: 'missing'",
        ),
    ],
)
def test_missing_library_reference_uses_exact_error(
    spec: ReferenceSpec,
    value: ReferenceValue,
    expected: str,
) -> None:
    schema = _schema({"ref": spec}, {"ref": value})

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, _library())

    assert str(exc_info.value) == expected


def test_library_reference_without_library_uses_exact_error() -> None:
    pulse = CfgSectionSpec(label="Pulse", fields={})
    schema = _schema(
        {"module": ReferenceSpec(kind="module", allowed=[pulse])},
        {"module": ReferenceValue("named", CfgSectionValue())},
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, None)

    assert str(exc_info.value) == (
        "Cannot resolve library reference 'named' without ModuleLibrary"
    )


def test_reference_kind_is_forwarded_as_opaque_id() -> None:
    shape = CfgSectionSpec(label="Asset", fields={})
    schema = _schema(
        {"asset": ReferenceSpec(kind="app-local/asset", allowed=[shape])},
        {"asset": ReferenceValue("named", CfgSectionValue())},
    )
    calls: list[tuple[str, str]] = []

    def resolve_reference(kind: str, key: str, /) -> str | None:
        calls.append((kind, key))
        return "Asset"

    raw = lower_finished_cfg(
        schema,
        resolve_expression=None,
        resolve_reference=resolve_reference,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )

    assert raw == {"asset": {}}
    assert calls == [
        ("app-local/asset", "named"),
        ("app-local/asset", "named"),
    ]


def test_library_reference_unsupported_shape_uses_exact_error() -> None:
    direct_readout = CfgSectionSpec(label="Direct Readout", fields={})
    schema = _schema(
        {"module": ReferenceSpec(kind="module", allowed=[direct_readout])},
        {"module": ReferenceValue("drive", CfgSectionValue())},
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(
            schema,
            None,
            _library(modules={"drive": {"type": "pulse"}}),
        )

    assert str(exc_info.value) == (
        "Library reference 'drive' resolved to unsupported spec 'Pulse'; "
        "allowed labels: Direct Readout"
    )


@pytest.mark.parametrize(
    ("chosen_key", "expected"),
    [
        ("<Custom:Pulse", "Invalid custom reference key: '<Custom:Pulse'"),
        (
            "<Custom:Unknown>",
            "Unknown custom reference label 'Unknown'; allowed labels: Pulse",
        ),
    ],
)
def test_invalid_custom_reference_uses_exact_error(
    chosen_key: str, expected: str
) -> None:
    pulse = CfgSectionSpec(label="Pulse", fields={})
    schema = _schema(
        {"module": ReferenceSpec(kind="module", allowed=[pulse])},
        {"module": ReferenceValue(chosen_key, CfgSectionValue())},
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, None)

    assert str(exc_info.value) == expected


def test_shared_ports_preserve_stage_order_without_reference_caching() -> None:
    pulse = CfgSectionSpec(
        label="Pulse",
        fields={"gain": ScalarSpec("Gain", float)},
    )
    schema = _schema(
        {"module": ReferenceSpec(kind="module", allowed=[pulse])},
        {
            "module": ReferenceValue(
                "drive",
                CfgSectionValue(fields={"gain": EvalValue("gain", resolved=0.25)}),
            )
        },
    )
    calls: list[str] = []

    def resolve_reference(kind: str, key: str, /) -> str | None:
        calls.append(f"reference:{kind}:{key}")
        return "Pulse"

    def resolve_expression(expr: str, /) -> int | float:
        calls.append(f"expression:{expr}")
        return 0.25

    raw = lower_finished_cfg(
        schema,
        resolve_expression=resolve_expression,
        resolve_reference=resolve_reference,  # type: ignore[arg-type]
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )

    assert raw == {"module": {"gain": 0.25}}
    assert calls == [
        "reference:module:drive",
        "reference:module:drive",
        "expression:gain",
        "reference:module:drive",
        "expression:gain",
    ]


def test_shared_range_port_receives_centered_edges() -> None:
    schema = _schema(
        {"sweep": CenteredSweepSpec("Sweep")},
        {"sweep": CenteredSweepValue(center=10.0, span=4.0, expts=5)},
    )
    calls: list[tuple[float, float, int]] = []

    def make_range(start: float, stop: float, /, *, expts: int) -> object:
        calls.append((start, stop, expts))
        return {"start": start, "stop": stop, "expts": expts}

    raw = lower_finished_cfg(
        schema,
        resolve_expression=None,
        resolve_reference=None,
        make_range=make_range,
    )

    assert raw == {"sweep": {"start": 8.0, "stop": 12.0, "expts": 5}}
    assert calls == [(8.0, 12.0, 5)]
