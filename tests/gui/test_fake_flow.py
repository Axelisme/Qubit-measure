"""Smoke test — full flow via FakeAdapter, no Qt, no hardware."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter, FakeAnalyzeParams
from zcu_tools.experiment.v2_gui.registry import ADAPTERS, register_all
from zcu_tools.gui.adapter import (
    AnalyzeRequest,
    DirectValue,
    RunRequest,
    WritebackRequest,
    schema_to_dict,
)
from zcu_tools.gui.registry import Registry


def _make_ctx():
    ctx = MagicMock()
    ctx.md = MagicMock()
    ctx.ml = MagicMock()
    ctx.soc = MagicMock()
    ctx.soccfg = MagicMock()
    ctx.ml.get_module.side_effect = lambda name, override=None: {"name": name}
    return ctx


def test_fake_adapter_full_flow():
    adapter = FakeAdapter()
    ctx = _make_ctx()

    # 1. make_default_cfg
    schema = adapter.make_default_cfg(ctx)
    assert schema is not None

    # 2. schema_to_dict
    cfg_dict = schema_to_dict(schema, ctx.ml)
    assert "reps" in cfg_dict
    assert cfg_dict["reps"] == 100
    assert "sweep" in cfg_dict

    # 3. run
    schema.value.fields["noise_scale"] = DirectValue(0.05)
    run_req = RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg)
    result = adapter.run(run_req, schema)
    assert isinstance(result.data, np.ndarray)
    assert len(result.data) == 11

    # 4. analyze
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=FakeAnalyzeParams(threshold=0.0),
            md=ctx.md,
            ml=ctx.ml,
            predictor=getattr(ctx, "predictor", None),
        )
    )
    assert isinstance(analyze_result.peak, float)
    assert analyze_result.figure is not None

    # 5. get_writeback_items
    items = adapter.get_writeback_items(
        WritebackRequest(run_result=result, analyze_result=analyze_result, ctx=ctx)
    )
    assert len(items) == 1
    assert items[0].key == "fake_peak"
    assert items[0].proposed_value == analyze_result.peak


def test_registry_register_all_and_create():
    reg = Registry()
    register_all(reg)
    assert reg.has("onetone/fake_freq")
    adapter = reg.create("onetone/fake_freq")
    from zcu_tools.experiment.v2_gui.adapters.onetone.fakefreq import FakeFreqAdapter

    assert isinstance(adapter, FakeFreqAdapter)


def test_all_adapters_in_registry_dict_are_registered():
    reg = Registry()
    register_all(reg)
    for name in ADAPTERS:
        assert reg.has(name), f"{name!r} missing from registry after register_all()"
