"""Characterization tests for finished-cfg lowering behavior."""

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
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    lower_finished_cfg,
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


def test_lowers_scalar_literal_optional_section_and_device() -> None:
    schema = _schema(
        {
            "literal": LiteralSpec("fixed"),
            "count": ScalarSpec("Count", int),
            "optional": ScalarSpec("Optional", float, optional=True),
            "section": CfgSectionSpec(fields={"enabled": ScalarSpec("Enabled", bool)}),
            "device": DeviceRefSpec("Device"),
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
        {"module": ModuleRefSpec(allowed=[pulse])},
        {
            "module": ModuleRefValue(
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
        {"module": ModuleRefSpec(allowed=[pulse], optional=True)},
        {"module": None},
    )

    assert schema_to_raw_dict(schema, None, None) == {}


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        (
            ModuleRefSpec(allowed=[CfgSectionSpec(label="Pulse", fields={})]),
            ModuleRefValue("missing", CfgSectionValue()),
            "Unknown module reference: 'missing'",
        ),
        (
            WaveformRefSpec(allowed=[CfgSectionSpec(label="Const", fields={})]),
            WaveformRefValue("missing", CfgSectionValue()),
            "Unknown waveform reference: 'missing'",
        ),
    ],
)
def test_missing_library_reference_uses_exact_error(
    spec: ModuleRefSpec | WaveformRefSpec,
    value: ModuleRefValue | WaveformRefValue,
    expected: str,
) -> None:
    schema = _schema({"ref": spec}, {"ref": value})

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, _library())

    assert str(exc_info.value) == expected


def test_library_reference_without_library_uses_exact_error() -> None:
    pulse = CfgSectionSpec(label="Pulse", fields={})
    schema = _schema(
        {"module": ModuleRefSpec(allowed=[pulse])},
        {"module": ModuleRefValue("named", CfgSectionValue())},
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, None)

    assert str(exc_info.value) == (
        "Cannot resolve library reference 'named' without ModuleLibrary"
    )


def test_library_reference_unsupported_shape_uses_exact_error() -> None:
    direct_readout = CfgSectionSpec(label="Direct Readout", fields={})
    schema = _schema(
        {"module": ModuleRefSpec(allowed=[direct_readout])},
        {"module": ModuleRefValue("drive", CfgSectionValue())},
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
        {"module": ModuleRefSpec(allowed=[pulse])},
        {"module": ModuleRefValue(chosen_key, CfgSectionValue())},
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
        {"module": ModuleRefSpec(allowed=[pulse])},
        {
            "module": ModuleRefValue(
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
