"""Tests for SpectrumStore + SelectionService."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.services.alignment import PointsService
from zcu_tools.gui.app.fluxdep.services.store import SelectionService, SpectrumStore
from zcu_tools.gui.app.fluxdep.state import (
    SELECTION_VERSION_KEY,
    FluxDepState,
    SpectrumEntry,
)
from zcu_tools.notebook.persistance import PointsData, SpectrumData


def _empty_points() -> PointsData:
    e = np.empty(0, dtype=np.float64)
    return PointsData(dev_values=e.copy(), fluxs=e.copy(), freqs=e.copy())


def _entry(name: str) -> SpectrumEntry:
    e = np.linspace(0.0, 1.0, 3).astype(np.float64)
    raw = SpectrumData(
        dev_values=e.copy(),
        fluxs=e.copy(),
        freqs=e.copy(),
        signals=np.zeros((3, 3), dtype=np.complex128),
    )
    return SpectrumEntry(
        name=name, spec_type="OneTone", raw=raw, points=_empty_points()
    )


# --- SpectrumStore ---------------------------------------------------------


def test_store_list_get_remove():
    st = FluxDepState()
    store = SpectrumStore(st)
    st.put_spectrum(_entry("a"))
    st.put_spectrum(_entry("b"))
    assert set(store.list_spectrums()) == {"a", "b"}
    assert store.get_spectrum("a").name == "a"
    store.remove_spectrum("a")
    assert store.list_spectrums() == ["b"]


def test_store_set_active():
    st = FluxDepState()
    st.put_spectrum(_entry("a"))
    SpectrumStore(st).set_active("a")
    assert st.active_spectrum == "a"


# --- SelectionService ------------------------------------------------------


def _with_points(st: FluxDepState, name: str, devs, freqs) -> None:
    st.put_spectrum(_entry(name))
    PointsService(st).set_points(name, np.array(devs), np.array(freqs))


def test_derive_pointcloud_concatenates_in_order():
    st = FluxDepState()
    _with_points(st, "a", [0.0, 1.0], [5.0, 5.1])
    _with_points(st, "b", [2.0], [5.2])
    sel = SelectionService(st)
    fluxs, freqs = sel.derive_pointcloud()
    assert fluxs.shape == (3,)
    np.testing.assert_allclose(freqs, [5.0, 5.1, 5.2])


def test_derive_pointcloud_empty():
    st = FluxDepState()
    fluxs, freqs = SelectionService(st).derive_pointcloud()
    assert fluxs.size == 0 and freqs.size == 0


def test_set_selection_bumps_and_stores():
    st = FluxDepState()
    _with_points(st, "a", [0.0, 1.0, 2.0], [5.0, 5.1, 5.2])
    sel = SelectionService(st)
    sel.set_selection(np.array([True, False, True]))
    assert st.version.get(SELECTION_VERSION_KEY) == 1
    assert st.selection.selected is not None


def test_set_selection_length_mismatch_raises():
    st = FluxDepState()
    _with_points(st, "a", [0.0, 1.0], [5.0, 5.1])
    with pytest.raises(ValueError):
        SelectionService(st).set_selection(np.array([True, False, True]))  # 3 != 2
