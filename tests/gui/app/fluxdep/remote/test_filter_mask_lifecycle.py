"""Filter identity survives other spectra's edits at the search/RPC seams."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.search import ParamBounds
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpectrumEntry

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


def _entry(name: str, freqs: list[float]) -> SpectrumEntry:
    devs = np.linspace(0.0, 1.0, len(freqs))
    axis = np.linspace(0.0, 1.0, 5)
    return SpectrumEntry(
        name=name,
        spec_type="TwoTone",
        raw={
            "dev_values": axis.copy(),
            "fluxs": axis,
            "freqs": np.array([4.0, 8.0, 12.0]),
            "signals": np.ones((5, 3), dtype=np.complex128),
        },
        points={"dev_values": devs.copy(), "fluxs": devs, "freqs": np.array(freqs)},
        flux_half=0.5,
        flux_int=1.0,
        aligned=True,
        points_completed=True,
    )


def _processed(path: Path, name: str) -> str:
    ctrl = Controller(FluxDepState())
    try:
        ctrl.state.put_spectrum(_entry(name, [10.0, 11.0, 12.0]))
        return ctrl.export_spectrums(str(path))
    finally:
        ctrl.interactive.dispose()


@pytest.mark.parametrize(
    ("change", "expected_mask", "expected_freqs", "expected_fluxs"),
    [
        ("reset", [True, True, True, False, True], [10, 11, 12, 7], [0, 0.5, 1, 1]),
        ("set", [True, True, True, False, True], [10, 11, 12, 7], [0, 0.5, 1, 1]),
        ("restore", [False, True, True, True, True], [7, 10, 11, 12], [1, 0, 0.5, 1]),
        (
            "remove_load",
            [False, True, True, True, True],
            [7, 10, 11, 12],
            [1, 0, 0.5, 1],
        ),
        (
            "resize_b",
            [True, True, False, True, True, True],
            [4, 5, 8, 9, 10],
            [0, 0.5, 0, 0.5, 1],
        ),
        (
            "align_then_set_b",
            [True, True, False, True, True],
            [4, 5, 8, 9],
            [0.5, 0.75, 0.3, 0.7],
        ),
    ],
)
def test_filter_masks_follow_spectrum_points_not_joint_positions(
    route_harness: RouteHarness,
    tmp_path: Path,
    change: str,
    expected_mask: list[bool],
    expected_freqs: list[float],
    expected_fluxs: list[float],
) -> None:
    ctrl = route_harness.ctrl
    ctrl.state.put_spectrum(_entry("A", [4.0, 5.0, 6.0]))
    ctrl.state.put_spectrum(_entry("B", [6.0, 7.0]))
    ctrl.set_selection(np.array([True, True, False, False, True]), 0.08)
    ctrl.set_fit_params(
        "synthetic.hdf5",
        ParamBounds(EJ=(2, 15), EC=(0.2, 2), EL=(0.1, 2)),
        {},
        None,
        None,
    )
    client = route_harness.client()

    if change in ("reset", "set"):
        if change == "reset":
            ctrl.reset_points("A")
        ctrl.set_points("A", np.array([0.0, 0.5, 1.0]), np.array([10.0, 11.0, 12.0]))
    elif change == "restore":
        ctrl.remove_spectrum("A")
        ctrl.load_processed_spectrums(_processed(tmp_path / "restore.hdf5", "A"))
    elif change == "remove_load":
        ctrl.remove_spectrum("A")
        ctrl.load_processed_spectrums(_processed(tmp_path / "new.hdf5", "C"))
    elif change == "resize_b":
        ctrl.set_points("B", np.array([0.0, 0.5, 1.0]), np.array([8.0, 9.0, 10.0]))
    else:
        ctrl.set_alignment("A", 0.0, 1.0)
        aligned = ctrl.capture_search()
        np.testing.assert_array_equal(aligned.freqs, [4, 5, 7])
        np.testing.assert_array_equal(aligned.fluxs, [0.5, 0.75, 1])
        ctrl.set_points("B", np.array([0.3, 0.7]), np.array([8.0, 9.0]))

    snapshot = route_harness.request(client, "selection.snapshot")
    assert snapshot["ok"] is True, snapshot
    assert snapshot["result"] == {
        "selected": expected_mask,
        "min_distance": 0.08,
        "pending": False,
    }
    inputs = ctrl.capture_search()
    np.testing.assert_array_equal(inputs.freqs, expected_freqs)
    np.testing.assert_array_equal(inputs.fluxs, expected_fluxs)
