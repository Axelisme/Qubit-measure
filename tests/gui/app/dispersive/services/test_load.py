"""Tests for dispersive LoadService — onetone hdf5 → OnetoneEntry (GHz)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.dispersive.services.load import LoadService
from zcu_tools.gui.app.dispersive.state import DispersiveState, FluxoniumInputs


def _inputs() -> FluxoniumInputs:
    return FluxoniumInputs(
        params=(4.0, 1.0, 0.5),
        flux_half=0.5,
        flux_int=1.0,
        flux_period=2.0,
        bare_rf_seed=5.3,
    )


def test_load_onetone_requires_fit_inputs(onetone_hdf5):
    path, *_ = onetone_hdf5
    st = DispersiveState()
    with pytest.raises(RuntimeError, match="fit inputs"):
        LoadService(st).load_onetone(path)


def test_load_onetone_writes_entry_in_ghz(onetone_hdf5):
    path, dev_values, freqs_ghz, signals = onetone_hdf5
    st = DispersiveState()
    st.set_fit_inputs(_inputs())

    name = LoadService(st).load_onetone(path)

    assert name == "R1_flux_1.hdf5"
    assert st.onetone is not None
    raw = st.onetone.raw
    np.testing.assert_allclose(raw["freqs"], freqs_ghz)  # GHz, not Hz
    np.testing.assert_allclose(raw["dev_values"], dev_values)
    assert raw["signals"].shape == signals.shape


def test_load_onetone_derives_flux_from_alignment(onetone_hdf5):
    path, dev_values, _freqs, _signals = onetone_hdf5
    st = DispersiveState()
    st.set_fit_inputs(_inputs())  # flux_half=0.5, flux_period=2.0

    LoadService(st).load_onetone(path)

    assert st.onetone is not None
    raw = st.onetone.raw
    expected = (dev_values - 0.5) / 2.0 + 0.5  # value2flux
    np.testing.assert_allclose(raw["fluxs"], expected)
