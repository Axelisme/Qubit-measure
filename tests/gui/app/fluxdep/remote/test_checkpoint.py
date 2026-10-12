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
            source.state.put_spectrum(SpectrumEntry(name, "TwoTone", raw, PointsData(dev_values=np.empty(0), fluxs=np.empty(0), freqs=np.empty(0))))
            source.set_alignment(name, 0.0, 1.0)
            source.set_points(name, np.array([0.0]), np.array([4.0]))
        filepath = source.export_spectrums(str(tmp_path / "checkpoint.hdf5"))
    finally:
        source.interactive.dispose()
    for name in ["existing-a", "existing-b"]:
        route_harness.ctrl.state.put_spectrum(SpectrumEntry(name, "OneTone", raw, PointsData(dev_values=np.empty(0), fluxs=np.empty(0), freqs=np.empty(0))))
    client = route_harness.client()
    before = route_harness.request(client, "spectrum.list")["result"]
    reply = route_harness.request(client, "spectrum.load_processed", filepath=filepath)
    assert reply["ok"] is False
    assert "existing-a" in reply["error"]["message"]
    assert "existing-b" in reply["error"]["message"]
    assert route_harness.request(client, "spectrum.list")["result"] == before
