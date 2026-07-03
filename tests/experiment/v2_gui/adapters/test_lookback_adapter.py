from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.lookback import LookbackAdapter
from zcu_tools.gui.app.main.adapter import CfgSectionValue, DirectValue, ModuleRefValue
from zcu_tools.meta_tool import MetaDict


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.ml = MagicMock()
    ctx.ml.modules = {}
    ctx.ml.waveforms = {}
    ctx.md = MetaDict()
    ctx.qub_name = "Q1"
    return ctx


def test_lookback_default_ro_length_matches_notebook_seed() -> None:
    schema = LookbackAdapter().make_default_cfg(_make_ctx())

    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ModuleRefValue)
    ro_cfg = readout.value.fields["ro_cfg"]
    assert isinstance(ro_cfg, CfgSectionValue)
    assert ro_cfg.fields["ro_length"] == DirectValue(1.5)
