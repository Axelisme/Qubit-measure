from __future__ import annotations

import json
from pathlib import Path

import pytest
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    SavePaths,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
)
from zcu_tools.gui.services.session_persistence import (
    SESSION_VERSION,
    PersistedSession,
    PersistedTab,
    SessionPersistenceError,
    SessionPersistenceService,
)


def test_session_persistence_save_and_load_roundtrip(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    session = PersistedSession(
        version=SESSION_VERSION,
        active_tab_index=0,
        tabs=[
            PersistedTab(
                adapter_name="fake",
                cfg_raw={"x": 1, "sweep": {"start": 0.0, "stop": 1.0, "expts": 11}},
                save_paths_override=SavePaths("/tmp/data.h5", "/tmp/img.png"),
            )
        ],
    )

    svc.save_session(session)
    loaded = svc.load_session()

    assert loaded is not None
    assert loaded.version == SESSION_VERSION
    assert loaded.active_tab_index == 0
    assert len(loaded.tabs) == 1
    assert loaded.tabs[0].adapter_name == "fake"
    assert loaded.tabs[0].cfg_raw["x"] == 1
    assert loaded.tabs[0].save_paths_override == SavePaths(
        "/tmp/data.h5", "/tmp/img.png"
    )


def test_session_persistence_missing_file_returns_none(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    assert svc.load_session() is None


def test_session_persistence_rejects_legacy_sweep_step_none(tmp_path: Path):
    from zcu_tools.gui.adapter import (
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        SweepSpec,
    )

    svc = SessionPersistenceService(cache_dir=tmp_path)
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"sweep": SweepSpec(label="Sweep")}),
        value=CfgSectionValue(fields={}),
    )

    with pytest.raises(SessionPersistenceError, match="Sweep step is required"):
        svc.raw_to_schema(
            base,
            {"sweep": {"start": 0.0, "stop": 1.0, "expts": 11, "step": None}},
        )


def test_session_persistence_rejects_legacy_sweep_eval_edges(tmp_path: Path):
    from zcu_tools.gui.adapter import (
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        SweepSpec,
    )

    svc = SessionPersistenceService(cache_dir=tmp_path)
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"sweep": SweepSpec(label="Sweep")}),
        value=CfgSectionValue(fields={}),
    )

    with pytest.raises(SessionPersistenceError, match="Legacy sweep"):
        svc.raw_to_schema(
            base,
            {
                "sweep": {
                    "start": "=r_f - 10",
                    "stop": "=r_f + 10",
                    "expts": 11,
                    "step": 2.0,
                }
            },
        )


def test_session_persistence_rejects_previous_cache_version(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    svc.session_path.parent.mkdir(parents=True, exist_ok=True)
    svc.session_path.write_text(
        json.dumps({"version": 1, "tabs": [], "active_tab_index": None}),
        encoding="utf-8",
    )

    with pytest.raises(SessionPersistenceError, match="Unsupported session version"):
        svc.load_session()


def test_session_persistence_roundtrip_preserves_eval_values(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "freq": ScalarSpec(label="Freq", type=float),
                "sweep": SweepSpec(label="Sweep"),
            }
        ),
        value=CfgSectionValue(
            fields={
                "freq": EvalValue(expr="r_f", resolved=6000.0, error=None),
                "sweep": SweepValue(
                    start=EvalValue(expr="r_f - rf_w", resolved=5980.0, error=None),
                    stop=EvalValue(expr="r_f + rf_w", resolved=6020.0, error=None),
                    expts=101,
                    step=0.4,
                ),
            }
        ),
    )

    raw = svc.schema_to_raw(schema, ml=None)
    restored = svc.raw_to_schema(
        CfgSchema(spec=schema.spec, value=CfgSectionValue(fields={})),
        raw,
    )

    freq = restored.value.fields["freq"]
    sweep = restored.value.fields["sweep"]
    assert isinstance(freq, EvalValue)
    assert freq.expr == "r_f"
    assert isinstance(sweep, SweepValue)
    assert isinstance(sweep.start, EvalValue)
    assert isinstance(sweep.stop, EvalValue)
    assert sweep.start.expr == "r_f - rf_w"
    assert sweep.stop.expr == "r_f + rf_w"


# ---------------------------------------------------------------------------
# WaveformRefSpec roundtrip
# ---------------------------------------------------------------------------


def test_session_persistence_waveform_ref_roundtrip(tmp_path: Path):
    inner_spec = CfgSectionSpec(
        fields={"width": ScalarSpec(label="Width", type=float)},
        label="Gaussian",
    )
    svc = SessionPersistenceService(cache_dir=tmp_path)
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "wf": WaveformRefSpec(allowed=[inner_spec], label="Waveform"),
            }
        ),
        value=CfgSectionValue(
            fields={
                "wf": WaveformRefValue(
                    chosen_key="Gaussian",
                    value=CfgSectionValue(fields={"width": DirectValue(value=50.0)}),
                ),
            }
        ),
    )

    raw = svc.schema_to_raw(schema, ml=None)
    restored = svc.raw_to_schema(
        CfgSchema(spec=schema.spec, value=CfgSectionValue(fields={})),
        raw,
    )

    wf = restored.value.fields["wf"]
    assert isinstance(wf, WaveformRefValue)
    assert wf.chosen_key == "Gaussian"
    assert wf.value.fields["width"] == DirectValue(value=50.0)
    # default (not overridden) round-trips as False
    assert wf.is_overridden is False


def test_session_persistence_waveform_ref_preserves_override(tmp_path: Path):
    """is_overridden=True survives the encode/decode round-trip (regression)."""
    inner_spec = CfgSectionSpec(
        fields={"width": ScalarSpec(label="Width", type=float)},
        label="Gaussian",
    )
    svc = SessionPersistenceService(cache_dir=tmp_path)
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"wf": WaveformRefSpec(allowed=[inner_spec], label="Waveform")}
        ),
        value=CfgSectionValue(
            fields={
                "wf": WaveformRefValue(
                    chosen_key="Gaussian",
                    value=CfgSectionValue(fields={"width": DirectValue(value=50.0)}),
                    is_overridden=True,
                ),
            }
        ),
    )

    raw = svc.schema_to_raw(schema, ml=None)
    restored = svc.raw_to_schema(
        CfgSchema(spec=schema.spec, value=CfgSectionValue(fields={})),
        raw,
    )

    wf = restored.value.fields["wf"]
    assert isinstance(wf, WaveformRefValue)
    assert wf.is_overridden is True


# ---------------------------------------------------------------------------
# Legacy payload rejection
# ---------------------------------------------------------------------------


def test_session_persistence_rejects_legacy_scalar_eval_expr(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"freq": ScalarSpec(label="Freq", type=float)}),
        value=CfgSectionValue(fields={}),
    )

    with pytest.raises(SessionPersistenceError, match="Legacy scalar"):
        svc.raw_to_schema(base, {"freq": "=r_f + 10"})


# ---------------------------------------------------------------------------
# DeviceRefSpec errors
# ---------------------------------------------------------------------------


def test_session_persistence_device_ref_value_must_be_string(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Device")}),
        value=CfgSectionValue(fields={}),
    )

    with pytest.raises(
        SessionPersistenceError, match="Device reference value must be string"
    ):
        svc.raw_to_schema(
            base,
            {"dev": {"__kind": "direct", "value": 123, "is_unset": False}},
        )


def test_session_persistence_device_ref_must_use_direct_encoding(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Device")}),
        value=CfgSectionValue(fields={}),
    )

    with pytest.raises(
        SessionPersistenceError, match="Device reference must use direct"
    ):
        svc.raw_to_schema(
            base,
            {"dev": "lo_device"},
        )


# ---------------------------------------------------------------------------
# active_tab_index validation
# ---------------------------------------------------------------------------


def test_session_persistence_rejects_non_integer_active_tab_index(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    svc.session_path.parent.mkdir(parents=True, exist_ok=True)
    svc.session_path.write_text(
        json.dumps(
            {
                "version": SESSION_VERSION,
                "tabs": [],
                "active_tab_index": "first",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        SessionPersistenceError, match="active_tab_index must be an integer"
    ):
        svc.load_session()


# ---------------------------------------------------------------------------
# save_paths_override validation
# ---------------------------------------------------------------------------


def test_session_persistence_rejects_non_dict_save_paths_override(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    svc.session_path.parent.mkdir(parents=True, exist_ok=True)
    svc.session_path.write_text(
        json.dumps(
            {
                "version": SESSION_VERSION,
                "tabs": [
                    {
                        "adapter_name": "fake",
                        "cfg_raw": {},
                        "save_paths_override": "not_a_dict",
                    }
                ],
                "active_tab_index": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        SessionPersistenceError, match="save_paths_override must be an object"
    ):
        svc.load_session()


# ---------------------------------------------------------------------------
# write_payload error handling
# ---------------------------------------------------------------------------


def test_session_persistence_write_error_raises_and_cleans_temp(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    session = PersistedSession(
        version=SESSION_VERSION,
        active_tab_index=None,
        tabs=[],
    )
    # Make the cache directory read-only to force an OSError during temp file write
    (tmp_path).chmod(0o444)
    try:
        with pytest.raises(SessionPersistenceError, match="Failed to save session"):
            svc.save_session(session)
    finally:
        (tmp_path).chmod(0o755)
