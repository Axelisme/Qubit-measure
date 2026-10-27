"""Explicit native selection, orientation and frequency conversion contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep import project_native_spectrum
from zcu_tools.datafile import Axis, Container, ContainerAttrs, Variable


@pytest.fixture
def native_container():
    signals = np.array([[1j, 2j], [3j, 4j], [5j, 6j]])
    variable = Variable(
        signals,
        (
            Axis("frequency", "Hz", np.array([4e9, 5e9, 6e9])),
            Axis("bias", "A", np.array([0.001, 0.002])),
        ),
        "a.u.",
    )
    return Container(
        vars={"chosen": variable, "also_complex": variable},
        docs={},
        groups={},
        attrs=ContainerAttrs("3.0", "test", None, {}),
    )


@pytest.mark.parametrize(
    "unit, scale", [("Hz", 1e9), ("kHz", 1e6), ("MHz", 1e3), ("GHz", 1.0)]
)
@pytest.mark.parametrize("device_first", [True, False])
def test_explicit_projection_handles_multiple_variables_and_axis_order(
    native_container, unit, scale, device_first
):
    variable = native_container.vars["chosen"]
    frequency = Axis("frequency", unit, np.array([4.0, 5.0, 6.0]) * scale)
    device = variable.axes[1]
    native_container = replace(
        native_container,
        vars={
            **native_container.vars,
            "chosen": Variable(
                variable.values.T if device_first else variable.values,
                (device, frequency) if device_first else (frequency, device),
                "a.u.",
            ),
        },
    )
    result = project_native_spectrum(
        native_container, variable="chosen", device_axis="bias"
    )
    assert result.unit == "A"
    np.testing.assert_array_equal(result.dev_values, [0.001, 0.002])
    np.testing.assert_array_equal(result.freqs_mhz, [4000, 5000, 6000])
    np.testing.assert_array_equal(result.signals, [[1j, 3j, 5j], [2j, 4j, 6j]])


@pytest.mark.parametrize(
    "invalid",
    [
        "complex_device",
        "complex_frequency",
        "zero_imaginary",
        "short_axis",
        "extra_axis",
        "matrix_axis",
    ],
)
def test_projection_rejects_nonreal_or_mismatched_axes_with_candidates(
    native_container, invalid
):
    chosen = native_container.vars["chosen"]
    axes = list(chosen.axes)
    if invalid in ("complex_device", "zero_imaginary"):
        axes[1] = axes[1]._replace(
            values=np.array([0.001, 0.002], dtype=complex)
            + (1j if invalid == "complex_device" else 0j)
        )
    elif invalid == "complex_frequency":
        axes[0] = axes[0]._replace(values=np.array([4e9, 5e9, 6e9], dtype=complex) + 1j)
    elif invalid == "short_axis":
        axes[1] = axes[1]._replace(values=np.array([0.001]))
    elif invalid == "extra_axis":
        axes.append(Axis("extra", "s", np.array([1.0])))
    else:
        axes[1] = axes[1]._replace(values=np.array([[0.001, 0.002]]))
    source = replace(
        native_container,
        vars={**native_container.vars, "chosen": replace(chosen, axes=tuple(axes))},
    )
    with pytest.raises(ValueError, match="available variables/axes") as caught:
        project_native_spectrum(source, variable="chosen", device_axis="bias")
    for candidate in ("chosen", "also_complex", "frequency [Hz]", "bias [A]"):
        assert candidate in str(caught.value)


@pytest.mark.parametrize("invalid", ["variable", "axis", "real", "1d", "unit"])
def test_invalid_projection_lists_all_variables_and_axis_units(
    native_container, invalid
):
    chosen = native_container.vars["chosen"]
    if invalid == "real":
        chosen = replace(chosen, values=chosen.values.real)
    elif invalid == "1d":
        chosen = Variable(np.array([1j, 2j]), (chosen.axes[1],), "a.u.")
    elif invalid == "unit":
        chosen = replace(
            chosen, axes=(chosen.axes[0]._replace(unit="hz"), chosen.axes[1])
        )
    source = replace(native_container, vars={**native_container.vars, "chosen": chosen})
    with pytest.raises(ValueError, match="available variables/axes") as caught:
        project_native_spectrum(
            source,
            variable="missing" if invalid == "variable" else "chosen",
            device_axis="missing" if invalid == "axis" else "bias",
        )
    message = str(caught.value)
    for candidate in ("chosen", "also_complex", "frequency [Hz]", "bias [A]"):
        assert candidate in message
