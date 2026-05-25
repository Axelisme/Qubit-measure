from __future__ import annotations

from pathlib import Path

import pytest
from zcu_tools.gui.adapter import SavePaths
from zcu_tools.gui.services.session_persistence import (
    PersistedSession,
    PersistedTab,
    SessionPersistenceService,
)


def test_session_persistence_save_and_load_roundtrip(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    session = PersistedSession(
        version=1,
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
    assert loaded.version == 1
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


def test_session_persistence_restores_legacy_sweep_step_none(tmp_path: Path):
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

    restored = svc.raw_to_schema(
        base,
        {"sweep": {"start": 0.0, "stop": 1.0, "expts": 11, "step": None}},
    )
    sweep = restored.value.fields["sweep"]
    assert sweep.step == pytest.approx(0.1)  # type: ignore[union-attr]


def test_session_persistence_restores_sweep_eval_edges(tmp_path: Path):
    from zcu_tools.gui.adapter import (
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        EvalValue,
        SweepSpec,
    )

    svc = SessionPersistenceService(cache_dir=tmp_path)
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"sweep": SweepSpec(label="Sweep")}),
        value=CfgSectionValue(fields={}),
    )

    restored = svc.raw_to_schema(
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
    sweep = restored.value.fields["sweep"]
    start = sweep.start  # type: ignore[union-attr]
    stop = sweep.stop  # type: ignore[union-attr]
    assert isinstance(start, EvalValue)
    assert isinstance(stop, EvalValue)
    assert start.expr == "=r_f - 10"
    assert stop.expr == "=r_f + 10"


def test_session_persistence_roundtrip_preserves_eval_values(tmp_path: Path):
    from zcu_tools.gui.adapter import (
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        EvalValue,
        ScalarSpec,
        SweepSpec,
        SweepValue,
    )

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
