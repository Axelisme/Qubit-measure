from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2.twotone.ro_optimize.auto_optimize import AutoOptCfg
from zcu_tools.experiment.v2.twotone.ro_optimize.freq import FreqCfg
from zcu_tools.experiment.v2.twotone.ro_optimize.freq_gain import FreqGainCfg
from zcu_tools.experiment.v2.twotone.ro_optimize.length import LengthCfg
from zcu_tools.experiment.v2.twotone.ro_optimize.power import PowerCfg
from zcu_tools.experiment.v2_gui.adapters.twotone.ro_optimize import (
    RoOptAutoAdapter,
    RoOptFreqAdapter,
    RoOptFreqGainAdapter,
    RoOptLengthAdapter,
    RoOptPowerAdapter,
)
from zcu_tools.gui.adapter import CfgSchema, RunRequest
from zcu_tools.meta_tool import MetaDict
from zcu_tools.program.v2 import SweepCfg


def _make_ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def _make_ctx(ml: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or _make_ml()
    ctx.md = MetaDict()
    ctx.qub_name = "Q1"
    return ctx


def _make_req(ml: MagicMock | None = None) -> RunRequest:
    return RunRequest(md=MagicMock(), ml=ml or _make_ml(), soc=None, soccfg=None)


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema.to_raw_dict(None, req.ml)


@pytest.mark.parametrize(
    ("adapter", "cfg_model", "sweep_key"),
    [
        (RoOptFreqAdapter(), FreqCfg, "freq"),
        (RoOptPowerAdapter(), PowerCfg, "gain"),
        (RoOptLengthAdapter(), LengthCfg, "length"),
    ],
)
def test_ro_opt_1d_build_exp_cfg_delegates(
    adapter: Any, cfg_model: type, sweep_key: str
) -> None:
    ml = _make_ml()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "modules" in raw
    modules = cast(dict[str, Any], raw["modules"])
    assert "readout" in modules
    assert "qub_pulse" in modules
    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep[sweep_key], SweepCfg)

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, cfg_model)


def test_ro_opt_freq_gain_build_exp_cfg_delegates() -> None:
    ml = _make_ml()
    adapter = RoOptFreqGainAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep["freq"], SweepCfg)
    assert isinstance(sweep["gain"], SweepCfg)

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, FreqGainCfg)


def test_ro_opt_auto_pops_num_points_before_make_cfg() -> None:
    # num_points is a GUI-only runtime arg (like onetone/power_dep's
    # earlystop_snr): it lives in the cfg spec but must be stripped before
    # ml.make_cfg, then passed to AutoOptExp().run(num_points=...).
    ml = _make_ml()
    adapter = RoOptAutoAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "num_points" in raw  # present in the lowered cfg
    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep["freq"], SweepCfg)
    assert isinstance(sweep["gain"], SweepCfg)
    assert isinstance(sweep["length"], SweepCfg)

    adapter.build_exp_cfg(raw, _make_req(ml))
    passed_raw = ml.make_cfg.call_args.args[0]
    assert "num_points" not in passed_raw  # stripped before make_cfg
    assert ml.make_cfg.call_args.args[1] is AutoOptCfg


def test_ro_opt_auto_num_points_validation() -> None:
    adapter = RoOptAutoAdapter()
    with pytest.raises(ValueError):
        adapter._num_points({"num_points": 0})  # non-positive
    with pytest.raises(ValueError):
        adapter._num_points({"num_points": "x"})  # not an int
    assert adapter._num_points({"num_points": 500}) == 500


@pytest.mark.parametrize(
    ("adapter", "wb_keys"),
    [
        (RoOptFreqAdapter(), ["best_ro_freq"]),
        (RoOptPowerAdapter(), ["best_ro_gain"]),
        (RoOptLengthAdapter(), ["best_ro_length"]),
        (RoOptFreqGainAdapter(), ["best_ro_freq", "best_ro_gain"]),
        (
            RoOptAutoAdapter(),
            ["best_ro_freq", "best_ro_gain", "best_ro_length"],
        ),
    ],
)
def test_ro_opt_writeback_targets(adapter: Any, wb_keys: list[str]) -> None:
    # Each adapter writes its optimized readout params back to MetaDict scalars.
    result = MagicMock()
    result.best_freq = 6100.0
    result.best_gain = 0.12
    result.best_length = 1.5
    req = MagicMock()
    req.analyze_result = result
    items = adapter.get_writeback_items(req)
    assert [it.target_name for it in items] == wb_keys
