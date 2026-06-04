"""Tests for ContextService.coerce_md_value (replaces inspect ast.literal_eval)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.event_bus import EventBus
from zcu_tools.gui.app.main.io_manager import IOManager
from zcu_tools.gui.app.main.services.context import ContextService, MdValueError
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_svc(md: MetaDict | None = None) -> ContextService:
    md = md if md is not None else MetaDict()
    state = State(
        ExpContext(
            md=md,
            ml=ModuleLibrary(),
            soc=None,
            soccfg=None,
            result_dir="",
            readiness=ContextReadiness.DRAFT,
        )
    )
    return ContextService(state, IOManager(), EventBus())


def test_coerce_new_key_accepts_int():
    svc = _make_svc()
    assert svc.coerce_md_value("freq", "100") == 100


def test_coerce_new_key_accepts_float():
    svc = _make_svc()
    assert svc.coerce_md_value("freq", "1.5") == pytest.approx(1.5)


def test_coerce_new_key_accepts_bool_lowercase():
    svc = _make_svc()
    assert svc.coerce_md_value("flag", "true") is True
    assert svc.coerce_md_value("flag", "False") is False


def test_coerce_new_key_accepts_bare_string():
    svc = _make_svc()
    assert svc.coerce_md_value("name", "hello world") == "hello world"


def test_coerce_existing_key_coerces_to_type():
    md = MetaDict()
    md.freq = 7000
    svc = _make_svc(md)
    assert svc.coerce_md_value("freq", "8000") == 8000
    assert isinstance(svc.coerce_md_value("freq", "8000"), int)


def test_coerce_existing_key_rejects_invalid_int():
    md = MetaDict()
    md.freq = 7000
    svc = _make_svc(md)
    with pytest.raises(MdValueError, match="Expected int"):
        svc.coerce_md_value("freq", "8.5")


def test_coerce_existing_key_rejects_invalid_bool():
    md = MetaDict()
    md.flag = True
    svc = _make_svc(md)
    with pytest.raises(MdValueError, match="Expected bool"):
        svc.coerce_md_value("flag", "maybe")


def test_coerce_existing_key_unknown_type_rejected():
    md = MetaDict()
    md.payload = [1, 2, 3]
    svc = _make_svc(md)
    with pytest.raises(MdValueError, match="Unsupported existing value type"):
        svc.coerce_md_value("payload", "ignored")


def test_coerce_no_context_falls_back_to_new_key_path():
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    # No startup context — service should treat key as new and accept scalars.
    svc = ContextService(state, IOManager(), EventBus())
    assert svc.coerce_md_value("anything", "42") == 42
