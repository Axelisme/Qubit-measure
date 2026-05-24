from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.v2.lookback import LookbackCfg
from zcu_tools.experiment.v2_gui.adapters.lookback import (
    LookbackAdapter,
    LookbackAnalyzeParams,
    LookbackRunResult,
)
from zcu_tools.gui.adapter import (
    AnalyzeRequest,
    CfgSchema,
    MetaDictWriteback,
    RunRequest,
    SaveDataRequest,
)
from zcu_tools.gui.adapter.lowering import schema_to_dict


def _make_ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def _make_ctx(ml: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or _make_ml()
    ctx.md = MagicMock(spec=[])
    ctx.res_name = "R1"
    return ctx


def _make_req(ml: MagicMock | None = None, *, with_soc: bool = False) -> RunRequest:
    return RunRequest(
        md=MagicMock(),
        ml=ml or _make_ml(),
        soc=MagicMock() if with_soc else None,
        soccfg=MagicMock() if with_soc else None,
    )


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema_to_dict(schema, req.ml)


def test_lookback_build_exp_cfg_delegates_to_ml_make_cfg() -> None:
    ml = _make_ml()
    adapter = LookbackAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    raw = _lower(schema, _make_req(ml))

    assert set(raw) == {"modules", "reps", "rounds", "relax_delay"}
    assert raw["reps"] == 1
    assert raw["rounds"] == 500
    modules = cast(dict[str, Any], raw["modules"])
    assert "readout" in modules
    assert "init_pulse" not in modules
    assert "reset" not in modules

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, LookbackCfg)


def test_lookback_run_without_soc_fast_fails() -> None:
    ml = _make_ml()
    adapter = LookbackAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)


def test_lookback_analyze_and_writeback(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = LookbackAdapter()
    cfg = MagicMock()
    cfg.modules.readout.ro_cfg = MagicMock()
    result = LookbackRunResult(
        times=np.array([0.0, 1.0]),
        signals=np.array([0.0 + 0.0j, 1.0 + 0.0j]),
        cfg_snapshot=cfg,
    )

    figure = MagicMock()

    def fake_analyze(self, raw_result, **kwargs):
        assert raw_result == (result.times, result.signals)
        assert kwargs["ratio"] == 0.1
        assert kwargs["smooth"] == 1.0
        assert kwargs["ro_cfg"] is cfg.modules.readout.ro_cfg
        return 0.42, figure

    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.adapters.lookback.LookbackExp.analyze",
        fake_analyze,
    )
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=LookbackAnalyzeParams(ratio=0.1, smooth=1.0, plot_fit=True),
            md=MagicMock(),
            ml=MagicMock(),
            predictor=None,
        )
    )
    assert analyze_result.predict_offset == 0.42
    assert analyze_result.figure is figure

    ctx = _make_ctx()
    items = adapter.get_writeback_items(
        MagicMock(run_result=result, analyze_result=analyze_result, ctx=ctx)
    )
    assert len(items) == 1
    assert items[0].key == "timeFly"
    item = cast(MetaDictWriteback, items[0])
    assert item.proposed_value == 0.42


def test_lookback_save_delegates_with_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_save_with_last_state(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.adapters.lookback.save_with_last_state",
        fake_save_with_last_state,
    )
    cfg = MagicMock()
    result = LookbackRunResult(
        times=np.array([0.0]),
        signals=np.array([1.0 + 0.0j]),
        cfg_snapshot=cfg,
    )

    LookbackAdapter().save(
        SaveDataRequest(
            run_result=result,
            data_path="/tmp/lookback",
            md=MagicMock(),
            ml=MagicMock(),
            chip_name="chip",
            qub_name="qub",
            res_name="res",
            active_label="label",
        )
    )

    assert captured["cfg"] is cfg
    assert captured["result"][0] is result.times
    assert captured["result"][1] is result.signals
    assert captured["filepath"] == "/tmp/lookback"
