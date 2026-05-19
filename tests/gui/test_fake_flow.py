"""Phase 4 smoke test — full flow via FakeAdapter, no Qt, no hardware."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import ADAPTERS, register_all
from zcu_tools.gui.adapter import schema_to_dict
from zcu_tools.gui.registry import Registry


def _make_ctx():
    ctx = MagicMock()
    ctx.ml = MagicMock()
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
    result = adapter.run(ctx, schema, noise_scale=0.05)
    assert isinstance(result, np.ndarray)
    assert len(result) == 11

    # 4. analyze
    analyze_result = adapter.analyze(result, ctx, threshold=0.0)
    peak, fig = analyze_result
    assert isinstance(peak, float)

    # 5. get_figure
    returned_fig = adapter.get_figure(analyze_result)
    assert returned_fig is fig

    # 6. get_writeback_spec
    items = adapter.get_writeback_spec(analyze_result, ctx)
    assert len(items) == 1
    assert items[0].key == "fake_peak"
    assert items[0].new_value == peak

    # 7. apply_writeback (no-op, just must not raise)
    adapter.apply_writeback(ctx, analyze_result, ["fake_peak"])


def test_registry_register_all_and_create():
    reg = Registry()
    register_all(reg)
    assert reg.has("fake")
    adapter = reg.create("fake")
    assert isinstance(adapter, FakeAdapter)


def test_all_adapters_in_registry_dict_are_registered():
    reg = Registry()
    register_all(reg)
    for name in ADAPTERS:
        assert reg.has(name), f"{name!r} missing from registry after register_all()"
