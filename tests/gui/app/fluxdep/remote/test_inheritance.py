"""Published alignment provenance survives drafts and changes at publication."""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


@pytest.fixture
def inherited_name(
    route_harness: RouteHarness, spectrum_hdf5, tmp_path: Path
) -> tuple[str, str]:
    """Load a second spectrum inheriting committed native alignment."""
    filepath, *_ = spectrum_hdf5
    ctrl = route_harness.ctrl
    source = ctrl.load_spectrum(filepath, "OneTone")
    ctrl.set_alignment(source, 0.0, 2.0)
    target = tmp_path / "child.hdf5"
    shutil.copyfile(filepath, target)
    client = route_harness.client()
    reply = route_harness.request(
        client,
        "spectrum.load",
        filepath=str(target),
        spec_type="OneTone",
        inherit_from=source,
    )
    assert reply["ok"] is True
    return "child.hdf5", source


def _snapshot(harness: RouteHarness, name: str) -> dict[str, object]:
    reply = harness.request(harness.links[0], "spectrum.snapshot", name=name)
    assert reply["ok"] is True
    return reply["result"]


def _inline_background(
    compute: Callable[[], object],
    on_done: Callable[[object], None],
    _on_error: Callable[[Exception], None],
) -> None:
    on_done(compute())


@pytest.mark.parametrize("action", ["unchanged", "manual", "auto", "discard", "direct"])
def test_alignment_source_is_published_with_alignment_not_with_draft(
    route_harness: RouteHarness,
    inherited_name: tuple[str, str],
    action: str,
) -> None:
    name, source = inherited_name
    ctrl = route_harness.ctrl
    initial = _snapshot(route_harness, name)
    assert initial["alignment"] == {"inherited_from": source}
    assert initial["pending"] is None
    assert initial["points"] == {"dev_values": [], "fluxs": [], "freqs": []}
    if action == "direct":
        ctrl.set_alignment(name, 0.1, 2.1)
    else:
        ctrl.set_active_spectrum(name)
        context = ctrl.interactive.begin_line_pick(name)
        context.plugin.bind_background(_inline_background)
        active = ctrl.interactive.inspect()
        assert active is not None
        params = {"name": name, "context_id": active.context_id}
        if action in ("manual", "discard"):
            command = route_harness.request(
                route_harness.links[0],
                "spectrum.interactive.command",
                **params,
                command="move_line",
                params={"role": "half", "position": 0.2},
            )
            assert command["ok"] is True
        elif action == "auto":
            command = route_harness.request(
                route_harness.links[0],
                "spectrum.interactive.command",
                **params,
                command="auto_align",
            )
            assert command["ok"] is True
        draft = _snapshot(route_harness, name)
        assert draft["alignment"] == {"inherited_from": source}
        assert draft["flux_half"] == initial["flux_half"]
        assert draft["pending"] == "align"
        terminal = route_harness.request(
            route_harness.links[0],
            "spectrum.interactive.command",
            **params,
            command="cancel" if action == "discard" else "finish",
        )
        assert terminal["ok"] is True
    published = _snapshot(route_harness, name)
    assert published["alignment"] == {
        "inherited_from": source if action in ("unchanged", "discard") else None
    }
    assert published["pending"] is None
    if action == "discard":
        assert published["flux_half"] == initial["flux_half"]
