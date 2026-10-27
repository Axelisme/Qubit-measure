"""Native input choices use real user input and visible dialog facts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from zcu_tools.analysis.flux import ComponentFluxContext, DevicePeriod, FluxFrame
from zcu_tools.datafile import Axis, load, save
from zcu_tools.gui.app.fluxdep.ui.load_dialog import LoadSpectrumDialog

from tests.gui._user_input import ComponentDriver
from tests.gui.app.fluxdep._native import native_spectrum_file


def test_native_unique_preselection_and_ambiguous_manual_selection(qapp, tmp_path):
    unique = native_spectrum_file(tmp_path / "unique.h5", None)
    container = load(unique)
    ambiguous = tmp_path / "ambiguous.h5"
    save(
        ambiguous,
        replace(container, vars={**container.vars, "second": container.vars["signal"]}),
    )
    dialog = LoadSpectrumDialog({}, start_dir=str(tmp_path))
    dialog.show()
    qapp.processEvents()
    driver = ComponentDriver(dialog)
    try:
        driver.click("Transpose axes")
        driver.screenshot(tmp_path / "before-native.png")
        driver.choose_file("Browse…", unique, evidence=tmp_path / "unique-chooser.png")
        driver.screenshot(tmp_path / "unique.png")
        assert not driver.observe("Transpose axes").checked
        assert not driver.observe("Transpose axes").enabled
        assert driver.observe(object_name="native_variable").text == "signal"
        assert driver.observe(object_name="native_device_axis").text == "bias"
        assert driver.observe(object_name="native_device").text == "unnamed_device"
        assert driver.observe("A").visible
        assert driver.observe("OK").enabled

        driver.choose_file(
            "Browse…", ambiguous, evidence=tmp_path / "ambiguous-chooser.png"
        )
        driver.screenshot(tmp_path / "ambiguous.png")
        assert driver.observe(object_name="native_variable").text == "(choose variable)"
        assert not driver.observe("OK").enabled
        driver.select_combo("second", object_name="native_variable")
        driver.screenshot(tmp_path / "manual.png")
        assert driver.observe(object_name="native_device_axis").text == "bias"
        assert driver.observe("OK").enabled
    finally:
        driver.click("Cancel")
        dialog.deleteLater()
        qapp.processEvents()


def test_native_axis_ambiguity_requires_explicit_choice_and_shows_context_error(
    qapp, tmp_path
):
    path = native_spectrum_file(tmp_path / "two-frequency-axes.h5", None)
    source = load(path)
    variable = source.vars["signal"]
    save(
        path,
        replace(
            source,
            vars={
                "signal": replace(
                    variable,
                    axes=(
                        Axis("bias", "GHz", np.array([0.001, 0.002])),
                        variable.axes[1],
                    ),
                )
            },
        ),
        replace=True,
    )
    dialog = LoadSpectrumDialog(
        {},
        start_dir=str(tmp_path),
        context_error="No component flux provider configured for component 'Q1' (kind 'qubit/fluxonium')",
    )
    dialog.show()
    qapp.processEvents()
    driver = ComponentDriver(dialog)
    try:
        driver.choose_file("Browse…", path, evidence=tmp_path / "chooser.png")
        driver.screenshot(tmp_path / "axis-required.png")
        assert driver.observe(object_name="native_device_axis").text == "(choose axis)"
        assert not driver.observe("OK").enabled
        assert "Q1" in driver.observe(object_name="native_input_error").text
        assert (
            "qubit/fluxonium" in driver.observe(object_name="native_input_error").text
        )
        driver.select_combo("bias", object_name="native_device_axis")
        driver.screenshot(tmp_path / "axis-selected.png")
        assert driver.observe("OK").enabled
        assert driver.observe(object_name="native_device").text == "unnamed_device"
    finally:
        driver.click("Cancel")
        dialog.deleteLater()
        qapp.processEvents()


def test_native_device_choice_uses_injected_default_and_units(qapp, tmp_path):
    path = native_spectrum_file(tmp_path / "data.h5", None)
    context = ComponentFluxContext(
        FluxFrame(
            0.25, {"bias": DevicePeriod(1.0, "A"), "coil": DevicePeriod(2.0, "A")}
        ),
        "bias",
        {"bias": "A", "coil": "A"},
    )
    dialog = LoadSpectrumDialog({}, start_dir=str(tmp_path), flux_context=context)
    dialog.show()
    qapp.processEvents()
    driver = ComponentDriver(dialog)
    try:
        driver.choose_file("Browse…", path, evidence=tmp_path / "chooser.png")
        driver.screenshot(tmp_path / "default-bias.png")
        assert driver.observe(object_name="native_device").text == "bias"
        assert driver.observe("OK").enabled
        driver.select_combo("coil", object_name="native_device")
        driver.screenshot(tmp_path / "selected-coil.png")
        assert driver.observe(object_name="native_device").text == "coil"
        assert driver.observe("A").visible
    finally:
        driver.click("Cancel")
        dialog.deleteLater()
        qapp.processEvents()
