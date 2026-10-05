"""Shared fixtures for fluxdep-gui tests."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs
from zcu_tools.analysis.fluxdep.twotone import TwoToneInputs
from zcu_tools.datafile import save_labber_data
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpectrumEntry


@pytest.fixture(autouse=True, name="_drain_qt_events")
def isolate_non_ui_qt_events() -> None:
    """Keep non-UI Fluxdep tests independent of QApplication."""


@pytest.fixture
def spectrum_hdf5(tmp_path):
    """Write a small 2D flux-dependence spectrum hdf5 via the native labber_io path.

    Returns (filepath, dev_values, freqs_GHz, signals) so a test can check the
    loaded arrays round-trip. The canonical layout is x=device values,
    y=frequency (Hz); ``save_labber_data`` stores z as native (Ny, Nx) with axes
    listed inner-first [inner=x=dev, outer=y=freq], so z = signals.T (the loader
    re-transposes with .T to recover device-major (Ndev, Nfreq)).
    """
    dev_values = np.linspace(-5.0, 5.0, 8).astype(np.float64)  # mA-like, increasing
    freqs_ghz = np.linspace(5.0, 6.0, 5).astype(np.float64)  # GHz, increasing
    rng = np.random.RandomState(0)
    signals = (
        rng.randn(len(dev_values), len(freqs_ghz))
        + 1j * rng.randn(len(dev_values), len(freqs_ghz))
    ).astype(np.complex128)

    filepath = str(tmp_path / "Q1_flux_1")
    # native (Ny, Nx) z, axes inner-first [inner=x=dev, outer=y=freq];
    # z = signals.T so the loader's .T recovers device-major signals.
    save_labber_data(
        filepath,
        z=("Signal", "a.u.", signals.T),
        axes=[
            ("Flux device value", "a.u.", dev_values),  # inner axis (x)
            ("Frequency", "Hz", freqs_ghz * 1e9),  # outer axis (y)
        ],
    )
    # save_labber_data forces the .hdf5 extension; return the resolved path.
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
    # legacy layout: inner=x=freq, outer=y=flux. native z is (Ny, Nx) =
    # (Nflux, Nfreq) == signals; the loader's .T then transpose_axes recovers
    # the canonical (flux, freq) == signals.
    save_labber_data(
        filepath,
        z=("Signal", "a.u.", signals),
        axes=[
            ("Frequency", "Hz", freqs_ghz * 1e9),  # inner axis (x)
            ("Flux device value", "a.u.", flux),  # outer axis (y)
        ],
    )
    return filepath + ".hdf5", flux, freqs_ghz, signals


@pytest.fixture
def twotone_inputs() -> TwoToneInputs:
    """Descending device rows with a sloped finite resonance in GHz."""
    devs = np.linspace(1.0, -1.0, 24)
    freqs = np.linspace(4.0, 6.0, 80)
    center = 4.8 + 0.3 * devs
    amplitude = np.exp(-(((freqs[None, :] - center[:, None]) / 0.08) ** 2))
    signals = np.asarray(1.0 + 1j * amplitude, dtype=np.complex128)
    return TwoToneInputs(FluxPickInputs(signals, devs, freqs))


@pytest.fixture
def twotone_controller(twotone_inputs: TwoToneInputs):
    """Aligned TwoTone owner with descending axes for publication sorting."""
    spectrum = twotone_inputs.spectrum
    state = FluxDepState()
    state.put_spectrum(
        SpectrumEntry(
            name="two",
            spec_type="TwoTone",
            raw={
                "signals": spectrum.signals.copy(),
                "dev_values": spectrum.dev_values.copy(),
                "freqs": spectrum.freqs.copy(),
                "fluxs": spectrum.dev_values.copy(),
            },
            points={
                "dev_values": np.empty(0),
                "freqs": np.empty(0),
                "fluxs": np.empty(0),
            },
        )
    )
    ctrl = Controller(state)
    ctrl.set_active_spectrum("two")
    ctrl.set_alignment("two", -0.5, 0.5)
    yield ctrl
    ctrl.interactive.dispose()


@pytest.fixture
def onetone_controller():
    """Aligned two-dip OneTone, descending devices to observe service sorting."""
    devs = np.linspace(1.0, 0.0, 40)
    freqs = np.linspace(5.0, 6.0, 20)
    profile = np.exp(-((freqs - freqs[10]) ** 2) / (2 * 0.08**2))
    depth = np.full(devs.size, 0.3)
    for center in (0.25, 0.75):
        depth += 0.5 * np.exp(-((devs - center) ** 2) / (2 * 0.03**2))
    signals = np.asarray(1 - depth[:, None] * profile[None, :], dtype=np.complex128)
    state = FluxDepState()
    state.put_spectrum(
        SpectrumEntry(
            name="one",
            spec_type="OneTone",
            raw={
                "signals": signals,
                "dev_values": devs,
                "freqs": freqs,
                "fluxs": devs.copy(),
            },
            points={
                "dev_values": np.empty(0),
                "freqs": np.empty(0),
                "fluxs": np.empty(0),
            },
        )
    )
    ctrl = Controller(state)
    ctrl.set_active_spectrum("one")
    ctrl.set_alignment("one", 0.2, 0.7)
    yield ctrl
    ctrl.interactive.dispose()
