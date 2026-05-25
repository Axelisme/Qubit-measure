from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepCfg
from zcu_tools.experiment.v2.onetone.freq import FreqCfg
from zcu_tools.experiment.v2.onetone.power_dep import PowerDepCfg
from zcu_tools.experiment.v2_gui.adapters.onetone.flux_dep import (
    NoAnalyzeParams as FluxNoAnalyzeParams,
)
from zcu_tools.experiment.v2_gui.adapters.onetone.flux_dep import (
    OneToneFluxDepAdapter,
    OneToneFluxDepRunResult,
)
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import OneToneFreqAdapter
from zcu_tools.experiment.v2_gui.adapters.onetone.power_dep import (
    NoAnalyzeParams as PowerNoAnalyzeParams,
)
from zcu_tools.experiment.v2_gui.adapters.onetone.power_dep import (
    OneTonePowerDepAdapter,
    OneTonePowerDepRunResult,
)
from zcu_tools.gui.adapter import AnalyzeRequest, CfgSchema, RunRequest
from zcu_tools.gui.adapter.lowering import schema_to_dict
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


def test_onetone_freq_build_exp_cfg_delegates_to_ml_make_cfg() -> None:
    ml = _make_ml()
    adapter = OneToneFreqAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert set(raw) == {"modules", "reps", "rounds", "relax_delay", "sweep"}
    sweep = cast(dict[str, Any], raw["sweep"])
    modules = cast(dict[str, Any], raw["modules"])
    assert isinstance(sweep["freq"], SweepCfg)
    assert "readout" in modules
    assert "reset" not in modules
    readout = cast(dict[str, Any], modules["readout"])
    ro_cfg = cast(dict[str, Any], readout["ro_cfg"])
    assert "gen_ch" not in ro_cfg

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, FreqCfg)


@pytest.mark.parametrize(
    ("adapter", "cfg_model"),
    [
        (OneTonePowerDepAdapter(), PowerDepCfg),
        (OneToneFluxDepAdapter(), FluxDepCfg),
    ],
)
def test_onetone_2d_build_exp_cfg_delegates_to_ml_make_cfg(adapter, cfg_model) -> None:
    ml = _make_ml()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "modules" in raw
    assert "sweep" in raw
    modules = cast(dict[str, Any], raw["modules"])
    assert "readout" in modules

    adapter.build_exp_cfg(raw, _make_req(ml))
    assert ml.make_cfg.call_args.args[1] is cfg_model


def test_power_dep_build_exp_cfg_strips_earlystop_snr() -> None:
    ml = _make_ml()
    adapter = OneTonePowerDepAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert raw["earlystop_snr"] == 0.0
    adapter.build_exp_cfg(raw, _make_req(ml))
    assert "earlystop_snr" not in ml.make_cfg.call_args.args[0]


def test_flux_dep_build_exp_cfg_converts_device_section() -> None:
    ml = _make_ml()
    adapter = OneToneFluxDepAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    adapter.build_exp_cfg(raw, _make_req(ml))
    cfg_raw = ml.make_cfg.call_args.args[0]
    assert cfg_raw["dev"] == {"flux_yoko": {"label": "flux_dev"}}


@pytest.mark.parametrize(
    "adapter",
    [OneToneFreqAdapter(), OneTonePowerDepAdapter(), OneToneFluxDepAdapter()],
)
def test_real_onetone_run_without_soc_fast_fails(adapter) -> None:
    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)
