"""Published selection projection through the real RPC route."""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


def test_selection_snapshot_reports_native_null_and_full_published_mask(
    route_harness: RouteHarness,
) -> None:
    client = route_harness.client()
    initial = route_harness.request(client, "selection.snapshot")
    assert initial["ok"] is True
    assert initial["result"] == {"selected": None, "min_distance": 0.0}
    state = route_harness.ctrl.state
    state.put_spectrum(
        SpectrumEntry(
            name="points",
            spec_type="TwoTone",
            raw={
                "dev_values": np.array([0.0, 1.0]),
                "fluxs": np.array([0.0, 1.0]),
                "freqs": np.array([4.0, 5.0]),
                "signals": np.ones((2, 2), dtype=np.complex128),
            },
            points={
                "dev_values": np.array([0.4, 0.8, 0.2]),
                "fluxs": np.array([0.4, 0.8, 0.2]),
                "freqs": np.array([4.4, 4.8, 4.2]),
            },
            aligned=True,
            points_completed=True,
        )
    )
    route_harness.ctrl.set_selection(np.array([False, True, False]), 0.35)
    before = state.version.snapshot()
    current = route_harness.request(client, "selection.snapshot")
    assert current["ok"] is True
    assert current["result"] == {"selected": [False, True, False], "min_distance": 0.35}
    assert state.version.snapshot() == before
    assert state.selection.selected is not None
    np.testing.assert_array_equal(state.selection.selected, [False, True, False])
