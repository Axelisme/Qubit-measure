"""ContextControlFacet delegation contract."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from zcu_tools.gui.session.context_control import ContextControlFacet


def _facet() -> tuple[ContextControlFacet, MagicMock, MagicMock]:
    context = MagicMock()
    device = MagicMock()
    return (
        ContextControlFacet(
            context=cast(Any, context),
            device=cast(Any, device),
        ),
        context,
        device,
    )


def test_context_control_facet_delegates_context_calls() -> None:
    facet, context, _device = _facet()

    context.has_project.return_value = True
    context.get_context_labels.return_value = ["ctx"]
    context.get_active_context_label.return_value = "ctx"
    context.list_value_sources.return_value = ("info",)
    context.read_value_source.return_value = ("info", 1.0)
    context.get_current_md.return_value = "md"
    context.get_current_ml.return_value = "ml"
    context.coerce_md_value.return_value = 2.0

    assert facet.has_project() is True
    facet.use_context("ctx")
    assert facet.get_context_labels() == ["ctx"]
    assert facet.get_active_context_label() == "ctx"
    assert facet.list_value_sources() == ("info",)
    assert facet.read_value_source("device.flux.value", "float") == ("info", 1.0)
    assert facet.get_current_md() == "md"
    assert facet.get_current_ml() == "ml"
    assert facet.coerce_md_value("r_f", "2.0") == 2.0
    facet.set_md_attr("r_f", 2.0)
    facet.del_md_attr("r_f")
    facet.rename_ml_module("readout", "readout2")
    facet.rename_ml_waveform("drive", "drive2")
    facet.del_ml_module("readout2")
    facet.del_ml_waveform("drive2")

    context.has_project.assert_called_once_with()
    context.use_context.assert_called_once_with("ctx")
    context.get_context_labels.assert_called_once_with()
    context.get_active_context_label.assert_called_once_with()
    context.list_value_sources.assert_called_once_with()
    context.read_value_source.assert_called_once_with("device.flux.value", "float")
    context.get_current_md.assert_called_once_with()
    context.get_current_ml.assert_called_once_with()
    context.coerce_md_value.assert_called_once_with("r_f", "2.0")
    context.set_md_attr.assert_called_once_with("r_f", 2.0)
    context.del_md_attr.assert_called_once_with("r_f")
    context.rename_ml_module.assert_called_once_with("readout", "readout2")
    context.rename_ml_waveform.assert_called_once_with("drive", "drive2")
    context.del_ml_module.assert_called_once_with("readout2")
    context.del_ml_waveform.assert_called_once_with("drive2")


def test_context_control_new_context_without_bind_device_is_unbound() -> None:
    facet, context, device = _facet()

    facet.new_context(clone_from="base")

    device.get_device_unit_strict.assert_not_called()
    device.get_device_value_for_new_context.assert_not_called()
    context.new_context.assert_called_once_with(
        value=None,
        unit="none",
        clone_from="base",
    )


def test_context_control_new_context_resolves_bound_device_unit_and_value() -> None:
    facet, context, device = _facet()
    device.get_device_unit_strict.return_value = "A"
    device.get_device_value_for_new_context.return_value = 0.005

    facet.new_context(bind_device="flux", clone_from="base")

    device.get_device_unit_strict.assert_called_once_with("flux")
    device.get_device_value_for_new_context.assert_called_once_with("flux")
    context.new_context.assert_called_once_with(
        value=0.005,
        unit="A",
        clone_from="base",
    )
