"""Unit tests for zcu_tools.gui.app.measure.registry."""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.app.measure.registry import Registry

from tests.gui._adapter_fakes import DummyAdapter as _DummyAdapter


def test_register_and_create():
    reg = Registry()
    reg.register("dummy", _DummyAdapter)
    adapter = reg.create("dummy")
    assert adapter is _DummyAdapter
    assert isinstance(adapter, Adapter)


def test_create_unknown_raises_key_error():
    reg = Registry()
    with pytest.raises(KeyError, match="not found"):
        reg.create("no_such")


def test_register_duplicate_raises_value_error():
    reg = Registry()
    reg.register("dummy", _DummyAdapter)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("dummy", _DummyAdapter)


def test_list_names():
    reg = Registry()
    reg.register("a", _DummyAdapter)
    reg.register("b", _DummyAdapter)
    names = reg.list_names()
    assert set(names) == {"a", "b"}


def test_has_returns_true_for_registered():
    reg = Registry()
    reg.register("dummy", _DummyAdapter)
    assert reg.has("dummy")
    assert not reg.has("other")


def test_create_returns_the_registered_value_each_time():
    reg = Registry()
    reg.register("dummy", _DummyAdapter)
    a1 = reg.create("dummy")
    a2 = reg.create("dummy")
    assert a1 is a2 is _DummyAdapter


def test_replacement_detaches_candidate_and_clear_disables_creation():
    live = Registry()
    live.register("old", _DummyAdapter)
    candidate = Registry()
    candidate.register("new", _DummyAdapter)
    candidate.validate()
    live.replace_from(candidate)
    candidate.clear()
    assert live.list_names() == ["new"]
    assert live.create("new") is _DummyAdapter
    live.clear()
    with pytest.raises(KeyError):
        live.create("new")


def test_registry_cannot_publish_itself():
    registry = Registry()
    with pytest.raises(ValueError, match="itself"):
        registry.replace_from(registry)
