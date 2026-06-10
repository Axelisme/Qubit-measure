"""Shared fixtures for fluxdep-gui tests."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.utils.datasaver import save_data


@pytest.fixture
def spectrum_hdf5(tmp_path):
    """Write a small 2D flux-dependence spectrum hdf5 via the real save_data path.

    Returns (filepath, dev_values, freqs_GHz, signals) so a test can check the
    loaded arrays round-trip. Mirrors how onetone.FluxDepExp.save lays out the
    x/y/z axes: x = device values, y = frequency in Hz, z = signals2D.T.
    """
    dev_values = np.linspace(-5.0, 5.0, 8).astype(np.float64)  # mA-like, increasing
    freqs_ghz = np.linspace(5.0, 6.0, 5).astype(np.float64)  # GHz, increasing
    rng = np.random.RandomState(0)
    signals = (
        rng.randn(len(dev_values), len(freqs_ghz))
        + 1j * rng.randn(len(dev_values), len(freqs_ghz))
    ).astype(np.complex128)

    filepath = str(tmp_path / "Q1_flux_1")
    save_data(
        filepath=filepath,
        x_info={"name": "Flux device value", "unit": "a.u.", "values": dev_values},
        y_info={"name": "Frequency", "unit": "Hz", "values": freqs_ghz * 1e9},
        z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
    )
    # save_data forces the .hdf5 extension; return the resolved path.
    return filepath + ".hdf5", dev_values, freqs_ghz, signals


@pytest.fixture
def transposed_spectrum_hdf5(tmp_path):
    """A legacy-layout spectrum: x = frequency (Hz), y = flux (the transpose of
    the canonical layout). Loading with transpose_axes=True should recover the
    canonical (x=flux, y=freq) arrays.

    Returns (filepath, flux_values, freqs_GHz, signals_flux_by_freq).
    """
    flux = np.linspace(-5.0, 5.0, 8).astype(np.float64)
    freqs_ghz = np.linspace(5.0, 6.0, 5).astype(np.float64)
    rng = np.random.RandomState(1)
    # canonical signals are (flux, freq); the legacy file stores them transposed
    signals = (
        rng.randn(len(flux), len(freqs_ghz)) + 1j * rng.randn(len(flux), len(freqs_ghz))
    ).astype(np.complex128)

    filepath = str(tmp_path / "legacy_flux_1")
    # legacy layout: x=freq(Hz), y=flux, z stored as (x, y)=(freq, flux) → save_data
    # wants z_info=(x-major,...).T, i.e. (flux, freq) == signals here.
    save_data(
        filepath=filepath,
        x_info={"name": "Frequency", "unit": "Hz", "values": freqs_ghz * 1e9},
        y_info={"name": "Flux device value", "unit": "a.u.", "values": flux},
        z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    )
    return filepath + ".hdf5", flux, freqs_ghz, signals
