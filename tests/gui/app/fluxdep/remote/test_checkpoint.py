"""Processed checkpoints preserve sampling and reject name collisions atomically."""

from pathlib import Path

import numpy as np
from zcu_tools.analysis.fluxdep.models import PointsData
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpectrumEntry

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


def test_restore_rejects_all_conflicting_names_without_publication(
    route_harness: RouteHarness, tmp_path: Path
) -> None:
    source = Controller(FluxDepState())
    raw: SpectrumData = {
        "dev_values": np.array([0.0, 1.0]),
        "fluxs": np.array([0.0, 1.0]),
        "freqs": np.array([4.0, 5.0]),
        "signals": np.ones((2, 2), dtype=np.complex128),
    }
    try:
        for name in ["new", "existing-a", "existing-b"]:
            source.state.put_spectrum(
                SpectrumEntry(
                    name,
                    "TwoTone",
                    raw,
                    PointsData(
                        dev_values=np.empty(0), fluxs=np.empty(0), freqs=np.empty(0)
                    ),
                )
            )
            source.set_alignment(name, 0.0, 1.0)
            source.set_points(name, np.array([0.0]), np.array([4.0]))
        filepath = source.export_spectrums(str(tmp_path / "checkpoint.hdf5"))
    finally:
        source.interactive.dispose()
    for name in ["existing-a", "existing-b"]:
        route_harness.ctrl.state.put_spectrum(
            SpectrumEntry(
                name,
                "OneTone",
                raw,
                PointsData(
                    dev_values=np.empty(0), fluxs=np.empty(0), freqs=np.empty(0)
                ),
            )
        )
    client = route_harness.client()
    initial = route_harness.request(client, "spectrum.list")
    assert initial["ok"] is True
    before = initial["result"]
    reply = route_harness.request(client, "spectrum.load_processed", filepath=filepath)
    assert reply["ok"] is False
    assert "existing-a" in reply["error"]["message"]
    assert "existing-b" in reply["error"]["message"]
    current = route_harness.request(client, "spectrum.list")
    assert current["ok"] is True
    assert current["result"] == before


def test_export_restore_preserves_all_sampled_points_without_filter_masks(
    route_harness: RouteHarness, cross_controller: Controller, tmp_path: Path
) -> None:
    cross_controller.set_selection(np.array([True, False, True, False]), 0.025)
    filepath = cross_controller.export_spectrums(str(tmp_path / "checkpoint.hdf5"))
    client = route_harness.client()
    restored = route_harness.request(
        client, "spectrum.load_processed", filepath=filepath
    )
    assert restored["ok"] is True, restored
    names = restored["result"]["names"]
    assert isinstance(names, list)
    assert set(names) == {"a", "empty", "b"}
    for name, count in [("a", 3), ("empty", 0), ("b", 1)]:
        entry = route_harness.ctrl.state.get_spectrum(name)
        assert entry.points_completed and entry.aligned
        assert entry.point_count == count
        assert entry.filter_mask is None
    assert route_harness.ctrl.state.selection.min_distance == 0.0
