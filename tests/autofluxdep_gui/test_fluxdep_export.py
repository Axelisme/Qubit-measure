"""Fluxdep export sidecar tests."""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.services.fluxdep_export import (
    export_qubit_freq_fluxdep_spectrum,
)
from zcu_tools.gui.app.fluxdep.services.load import LoadService
from zcu_tools.gui.app.fluxdep.state import FluxDepState


def test_qubit_freq_export_loads_as_fluxdep_spectrum(tmp_path):
    result = QubitFreqResult.allocate(
        np.array([1.0, 0.0], dtype=float),
        np.array([-1.0, 0.0, 1.0], dtype=float),
    )
    result.predict_freq[:] = [5001.0, 5000.0]
    result.signal[0] = [10.0, 11.0, 12.0]
    result.signal[1] = [20.0, 21.0, 22.0]

    filepath = export_qubit_freq_fluxdep_spectrum(result, tmp_path / "qf_export.hdf5")

    state = FluxDepState()
    name = LoadService(state).load_spectrum(filepath, spec_type="TwoTone")
    raw = state.spectrums[name].raw

    np.testing.assert_allclose(raw["dev_values"], [0.0, 1.0])
    np.testing.assert_allclose(raw["freqs"], [4.999, 5.0, 5.001, 5.002])
    assert raw["signals"].shape == (2, 4)
    # Fluxdep loader sorts the descending flux axis; row 0 is original flux 0.0.
    np.testing.assert_allclose(
        raw["signals"][0].real,
        [20.0, 21.0, 22.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        raw["signals"][1].real,
        [np.nan, 10.0, 11.0, 12.0],
        equal_nan=True,
    )
