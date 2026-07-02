"""ContextControlFacet public contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.value_lookup import ValueInfo

from tests.gui._control_fakes import CallLog, RecordedCall, call


class RecordingContext:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.md = object()
        self.ml = object()
        self.source = ValueInfo(
            key="device.flux.value",
            type_=float,
            owner="device.flux",
        )

    def has_project(self) -> bool:
        self._log.add("context", "has_project")
        return True

    def use_context(self, label: str) -> None:
        self._log.add("context", "use_context", label)

    def new_context(
        self,
        *,
        value: float | None,
        unit: str,
        clone_from: str | None,
    ) -> None:
        self._log.add(
            "context",
            "new_context",
            value=value,
            unit=unit,
            clone_from=clone_from,
        )

    def get_context_labels(self) -> list[str]:
        self._log.add("context", "get_context_labels")
        return ["ctx"]

    def get_active_context_label(self) -> str | None:
        self._log.add("context", "get_active_context_label")
        return "ctx"

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        self._log.add("context", "list_value_sources")
        return (self.source,)

    def read_value_source(
        self, key: str, type_name: str | None
    ) -> tuple[ValueInfo, float]:
        self._log.add("context", "read_value_source", key, type_name)
        return self.source, 1.0

    def get_current_md(self) -> object:
        self._log.add("context", "get_current_md")
        return self.md

    def get_current_ml(self) -> object:
        self._log.add("context", "get_current_ml")
        return self.ml

    def coerce_md_value(self, key: str, text: str) -> float:
        self._log.add("context", "coerce_md_value", key, text)
        return 2.0

    def set_md_attr(self, key: str, value: object) -> None:
        self._log.add("context", "set_md_attr", key, value)

    def del_md_attr(self, key: str) -> None:
        self._log.add("context", "del_md_attr", key)

    def rename_ml_module(self, old: str, new: str) -> None:
        self._log.add("context", "rename_ml_module", old, new)

    def rename_ml_waveform(self, old: str, new: str) -> None:
        self._log.add("context", "rename_ml_waveform", old, new)

    def del_ml_module(self, name: str) -> None:
        self._log.add("context", "del_ml_module", name)

    def del_ml_waveform(self, name: str) -> None:
        self._log.add("context", "del_ml_waveform", name)


class RecordingDevice:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.unit = "A"
        self.value = 0.005

    def get_device_unit_strict(self, name: str) -> str:
        self._log.add("device", "get_device_unit_strict", name)
        return self.unit

    def get_device_value_for_new_context(self, name: str) -> float | None:
        self._log.add("device", "get_device_value_for_new_context", name)
        return self.value


def _facet() -> tuple[ContextControlFacet, CallLog, RecordingContext, RecordingDevice]:
    log = CallLog()
    context = RecordingContext(log)
    device = RecordingDevice(log)
    return (
        ContextControlFacet(context=cast(Any, context), device=cast(Any, device)),
        log,
        context,
        device,
    )


def test_context_control_facet_forwards_deliberate_context_contract() -> None:
    facet, log, context, _device = _facet()
    source = ValueInfo(key="device.flux.value", type_=float, owner="device.flux")

    cases: tuple[tuple[str, Callable[[], object], object, RecordedCall], ...] = (
        (
            "has_project",
            facet.has_project,
            True,
            call("context", "has_project"),
        ),
        (
            "use_context",
            lambda: facet.use_context("ctx"),
            None,
            call("context", "use_context", "ctx"),
        ),
        (
            "get_context_labels",
            facet.get_context_labels,
            ["ctx"],
            call("context", "get_context_labels"),
        ),
        (
            "get_active_context_label",
            facet.get_active_context_label,
            "ctx",
            call("context", "get_active_context_label"),
        ),
        (
            "list_value_sources",
            facet.list_value_sources,
            (source,),
            call("context", "list_value_sources"),
        ),
        (
            "read_value_source",
            lambda: facet.read_value_source("device.flux.value", "float"),
            (source, 1.0),
            call("context", "read_value_source", "device.flux.value", "float"),
        ),
        (
            "get_current_md",
            facet.get_current_md,
            context.md,
            call("context", "get_current_md"),
        ),
        (
            "get_current_ml",
            facet.get_current_ml,
            context.ml,
            call("context", "get_current_ml"),
        ),
        (
            "coerce_md_value",
            lambda: facet.coerce_md_value("r_f", "2.0"),
            2.0,
            call("context", "coerce_md_value", "r_f", "2.0"),
        ),
        (
            "set_md_attr",
            lambda: facet.set_md_attr("r_f", 2.0),
            None,
            call("context", "set_md_attr", "r_f", 2.0),
        ),
        (
            "del_md_attr",
            lambda: facet.del_md_attr("r_f"),
            None,
            call("context", "del_md_attr", "r_f"),
        ),
        (
            "rename_ml_module",
            lambda: facet.rename_ml_module("readout", "readout2"),
            None,
            call("context", "rename_ml_module", "readout", "readout2"),
        ),
        (
            "rename_ml_waveform",
            lambda: facet.rename_ml_waveform("drive", "drive2"),
            None,
            call("context", "rename_ml_waveform", "drive", "drive2"),
        ),
        (
            "del_ml_module",
            lambda: facet.del_ml_module("readout2"),
            None,
            call("context", "del_ml_module", "readout2"),
        ),
        (
            "del_ml_waveform",
            lambda: facet.del_ml_waveform("drive2"),
            None,
            call("context", "del_ml_waveform", "drive2"),
        ),
    )

    for name, action, expected_result, _expected_call in cases:
        assert action() == expected_result, name

    assert log.calls == [expected_call for *_, expected_call in cases]


def test_context_control_new_context_without_bind_device_is_unbound() -> None:
    facet, log, _context, _device = _facet()

    facet.new_context(clone_from="base")

    assert log.calls == [
        call("context", "new_context", value=None, unit="none", clone_from="base"),
    ]


def test_context_control_new_context_resolves_bound_device_unit_and_value() -> None:
    facet, log, _context, _device = _facet()

    facet.new_context(bind_device="flux", clone_from="base")

    assert log.calls == [
        call("device", "get_device_unit_strict", "flux"),
        call("device", "get_device_value_for_new_context", "flux"),
        call("context", "new_context", value=0.005, unit="A", clone_from="base"),
    ]
