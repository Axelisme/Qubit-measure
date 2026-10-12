"""Native spectrum evidence and unfinished draft facts through RPC routes."""

import base64
import struct

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


@pytest.fixture
def spectrum(route_harness: RouteHarness) -> SpectrumEntry:
    """Publish a small calibrated spectrum with two sampled points."""
    entry = SpectrumEntry(
        name="two",
        spec_type="TwoTone",
        raw={
            "dev_values": np.linspace(-1.0, 1.0, 5),
            "fluxs": np.linspace(0.0, 1.0, 5),
            "freqs": np.linspace(4.0, 5.0, 5),
            "signals": np.ones((5, 5), dtype=np.complex128),
        },
        points={
            "dev_values": np.array([-0.5, 0.5]),
            "fluxs": np.array([0.25, 0.75]),
            "freqs": np.array([4.2, 4.8]),
        },
        aligned=True,
        points_completed=True,
        alignment_seeded=True,
        flux_half=0.0,
        flux_int=1.0,
        flux_period=2.0,
    )
    route_harness.ctrl.state.put_spectrum(entry)
    return entry


def test_render_returns_png_without_mutating_published_state(
    route_harness: RouteHarness, spectrum: SpectrumEntry
) -> None:
    client = route_harness.client()
    before = route_harness.ctrl.state.version.snapshot()
    reply = route_harness.request(client, "spectrum.render", name=spectrum.name)
    assert reply["ok"], reply
    result = reply["result"]
    encoded = result["png_b64"]
    assert isinstance(encoded, str)
    png = base64.b64decode(encoded, validate=True)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert width > 0 and height > 0
    assert result["bytes"] == len(png)
    assert route_harness.ctrl.state.version.snapshot() == before


@pytest.mark.parametrize("draft", [False, True])
def test_list_reports_unfinished_alignment_draft(
    route_harness: RouteHarness, spectrum: SpectrumEntry, draft: bool
) -> None:
    client = route_harness.client()
    route_harness.ctrl.set_active_spectrum(spectrum.name)
    if draft:
        route_harness.ctrl.interactive.begin_line_pick(spectrum.name)
    reply = route_harness.request(client, "spectrum.list")
    assert reply["ok"], reply
    rows = reply["result"]["spectrums"]
    assert isinstance(rows, list)
    assert rows[0]["pending"] == ("align" if draft else None)


@pytest.mark.parametrize("draft", [False, True])
def test_selection_snapshot_reports_unfinished_filter_draft(
    route_harness: RouteHarness, spectrum: SpectrumEntry, draft: bool
) -> None:
    client = route_harness.client()
    if draft:
        route_harness.ctrl.interactive.begin_cross_selection()
    reply = route_harness.request(client, "selection.snapshot")
    assert reply["ok"], reply
    assert reply["result"]["pending"] is draft
    # Draft inspection does not publish its temporary mask.
    assert reply["result"]["selected"] is None
