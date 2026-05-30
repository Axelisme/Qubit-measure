"""Adapter cfg_spec / make_default_value split must compose to make_default_cfg.

After splitting the adapter API into a static cfg_spec() and a context-driven
make_default_value(ctx), make_default_cfg must remain byte-identical to the
composition, for every registered adapter.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import ExpContext
from zcu_tools.gui.registry import Registry


def _registry() -> Registry:
    reg = Registry()
    register_all(reg)
    return reg


def _ctx() -> ExpContext:
    # Adapters read md via .get(...)/md_has_key; an empty mapping mock exercises
    # the "no metadata" default path. ml is only touched by value builders.
    md = MagicMock()
    md.get.return_value = None
    md.__contains__ = lambda self, k: False
    return ExpContext(md=md, ml=MagicMock(), soc=None, soccfg=None)


@pytest.mark.parametrize("name", _registry().list_names())
def test_make_default_cfg_equals_spec_plus_value(name: str):
    adapter = _registry().create(name)
    # make_default_value is a BaseAdapter member, not part of the framework
    # ExpAdapterProtocol; every registered adapter is a BaseAdapter subclass.
    assert isinstance(adapter, BaseAdapter)
    ctx = _ctx()
    cfg = adapter.make_default_cfg(ctx)
    assert adapter.cfg_spec() == cfg.spec
    assert adapter.make_default_value(ctx) == cfg.value


def test_cfg_spec_is_context_free():
    # cfg_spec() takes no ctx; calling it twice yields equal specs (pure).
    adapter = _registry().create("onetone/fake_freq")
    assert adapter.cfg_spec() == adapter.cfg_spec()
