"""Public state writes and resource versions for fluxdep-gui."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.models import PointsData
from zcu_tools.analysis.fluxdep.search import ParamBounds
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.fluxdep.state import (
    PROJECT_VERSION_KEY,
    SELECTION_VERSION_KEY,
    SPECTRUM_SET_VERSION_KEY,
    AlignmentPublication,
    FluxDepState,
    SpectrumEntry,
    SpecType,
    VersionTable,
    spectrum_version_key,
)
from zcu_tools.gui.expected_error import (
    ExpectedError,
    ExpectedErrorCategory,
    FailedPreconditionError,
    InvalidInputError,
)


def _empty_points() -> PointsData:
    e = np.empty(0, dtype=np.float64)
    return PointsData(dev_values=e.copy(), fluxs=e.copy(), freqs=e.copy())


def _make_entry(name: str, spec_type: SpecType = "OneTone") -> SpectrumEntry:
    e = np.linspace(0.0, 1.0, 3).astype(np.float64)
    raw = SpectrumData(
        dev_values=e.copy(),
        fluxs=e.copy(),
        freqs=e.copy(),
        signals=np.zeros((3, 3), dtype=np.complex128),
    )
    return SpectrumEntry(
        name=name, spec_type=spec_type, raw=raw, points=_empty_points()
    )


# ---------------------------------------------------------------------------
# Shared VersionTable's existing bump/forget behavior
# ---------------------------------------------------------------------------


def test_version_absent_is_zero():
    vt = VersionTable()
    assert vt.get("anything") == 0


def test_version_bump_monotonic():
    vt = VersionTable()
    assert vt.bump("k") == 1
    assert vt.bump("k") == 2
    assert vt.get("k") == 2


def test_version_drop_prefix_resets_to_zero():
    vt = VersionTable()
    vt.bump("spectrum:a")
    vt.bump("spectrum:ab")
    vt.bump("other")
    vt.drop_prefix("spectrum:a")
    assert vt.get("spectrum:a") == 0
    assert vt.get("spectrum:ab") == 0  # prefix match
    assert vt.get("other") == 1  # untouched


def test_version_snapshot_is_copy():
    vt = VersionTable()
    vt.bump("k")
    snap = vt.snapshot()
    snap["k"] = 99
    assert vt.get("k") == 1


# ---------------------------------------------------------------------------
# FluxDepState — spectrum collection + version bump↔drop
# ---------------------------------------------------------------------------


def test_put_spectrum_new_bumps_per_key_and_set():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    assert st.version.get(spectrum_version_key("a")) == 1
    assert st.version.get(SPECTRUM_SET_VERSION_KEY) == 1
    assert "a" in st.spectrums


def test_put_spectrum_replace_does_not_bump_set():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    st.put_spectrum(_make_entry("a"))  # replace existing
    assert st.version.get(spectrum_version_key("a")) == 2  # per-key still moves
    assert st.version.get(SPECTRUM_SET_VERSION_KEY) == 1  # set unchanged


@pytest.mark.parametrize("name", ["a", "new"])
def test_put_spectrum_rejects_filter_mask_without_mutation(name: str):
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    st.set_active("a")
    st.set_selection(np.empty(0, dtype=np.bool_), 0.2)
    stored_entry = st.get_spectrum("a")
    before_versions = st.version.snapshot()
    incoming = _make_entry(name)
    incoming.filter_mask = np.empty(0, dtype=np.bool_)
    incoming_mask = incoming.filter_mask

    with pytest.raises(ValueError, match="cannot carry a Filter mask"):
        st.put_spectrum(incoming)

    assert incoming.filter_mask is incoming_mask
    assert list(st.spectrums) == ["a"]
    assert st.get_spectrum("a") is stored_entry
    assert st.active_spectrum == "a"
    assert st.selection.min_distance == 0.2
    assert st.version.snapshot() == before_versions


def test_remove_spectrum_drops_key_bumps_set():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    st.remove_spectrum("a")
    assert "a" not in st.spectrums
    assert st.version.get(spectrum_version_key("a")) == 0  # dropped
    assert st.version.get(SPECTRUM_SET_VERSION_KEY) == 2  # add + remove


@pytest.mark.parametrize("name", ["foo", "譜:*", "a:b"])
def test_remove_spectrum_preserves_prefix_sibling(name: str):
    st = FluxDepState()
    sibling = _make_entry(name + "bar")
    st.put_spectrum(_make_entry(name))
    st.put_spectrum(sibling)
    st.put_spectrum(sibling)
    st.set_active(sibling.name)

    st.remove_spectrum(name)

    assert st.spectrums == {sibling.name: sibling}
    assert st.active_spectrum == sibling.name
    assert st.version.get(spectrum_version_key(name)) == 0
    assert st.version.get(spectrum_version_key(sibling.name)) == 2
    assert st.version.get(SPECTRUM_SET_VERSION_KEY) == 3


def test_same_name_recreation_never_reuses_a_spectrum_version():
    st = FluxDepState()
    entry = _make_entry("a")
    key = spectrum_version_key(entry.name)
    st.put_spectrum(entry)
    assert st.version.get(key) == 1

    st.remove_spectrum(entry.name)
    assert st.version.get(key) == 0
    st.put_spectrum(_make_entry(entry.name))
    assert st.version.get(key) == 2

    st.remove_spectrum(entry.name)
    assert st.version.get(key) == 0
    st.put_spectrum(_make_entry(entry.name))
    assert st.version.get(key) == 3
    assert st.version.get(SPECTRUM_SET_VERSION_KEY) == 5


def test_remove_active_spectrum_clears_active():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    st.set_active("a")
    st.remove_spectrum("a")
    assert st.active_spectrum is None


def test_unknown_named_commands_leave_state_unchanged():
    st = FluxDepState()
    st.put_spectrum(_make_entry("譜:*"))
    st.set_active("譜:*")
    before = st.version.snapshot()
    entry = st.spectrums["譜:*"]
    for command in (
        lambda: st.get_spectrum("nope"),
        lambda: st.remove_spectrum("nope"),
        lambda: st.set_active("nope"),
        lambda: st.reset_alignment("nope"),
        lambda: st.reset_points("nope"),
    ):
        with pytest.raises(InvalidInputError) as caught:
            command()
        assert caught.value.category is ExpectedErrorCategory.INVALID_INPUT
        assert caught.value.reason_code == "unknown_spectrum"
        assert st.spectrums == {"譜:*": entry}
        assert st.active_spectrum == "譜:*"
        assert st.version.snapshot() == before


@pytest.mark.parametrize("name", ["譜:*", "a:b", "plain"])
def test_get_spectrum_returns_live_literal_entry(name: str):
    st = FluxDepState()
    entry = _make_entry(name)
    st.put_spectrum(entry)
    before = st.version.snapshot()
    assert st.get_spectrum(name) is entry
    assert st.version.snapshot() == before


def test_get_spectrum_rejects_foreign_thread_without_mutation():
    st = FluxDepState()
    entry = _make_entry("a")
    st.put_spectrum(entry)
    before = st.version.snapshot()
    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        pytest.raises(RuntimeError, match="owner|thread") as caught,
    ):
        # The submitted foreign-thread call must reject without any owner delivery.
        pool.submit(st.get_spectrum, "a").result(timeout=30)
    assert not isinstance(caught.value, ExpectedError)
    assert st.get_spectrum("a") is entry
    assert st.version.snapshot() == before


def test_reset_points_unaligned_is_correctable_without_mutation():
    st = FluxDepState()
    entry = _make_entry("a")
    st.put_spectrum(entry)
    before = st.version.snapshot()
    with pytest.raises(FailedPreconditionError) as caught:
        st.reset_points("a")
    assert caught.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert caught.value.reason_code == "spectrum_not_aligned"
    assert st.spectrums["a"] is entry
    assert st.version.snapshot() == before


def test_set_alignment_marks_aligned_and_bumps():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    v0 = st.version.get(spectrum_version_key("a"))
    new_fluxs = st.spectrums["a"].raw["dev_values"].copy()
    st.set_alignment(
        "a",
        alignment=AlignmentPublication(flux_half=1.0, flux_int=2.0, flux_period=2.0),
        new_fluxs=new_fluxs,
        new_point_fluxs=np.empty(0, dtype=np.float64),
    )
    entry = st.spectrums["a"]
    assert entry.aligned is True
    assert (entry.flux_half, entry.flux_int, entry.flux_period) == (1.0, 2.0, 2.0)
    assert st.version.get(spectrum_version_key("a")) == v0 + 1


def test_set_points_marks_selected_and_bumps():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    v0 = st.version.get(spectrum_version_key("a"))
    pts = PointsData(
        dev_values=np.array([1.0]),
        fluxs=np.array([0.5]),
        freqs=np.array([5.0]),
    )
    st.set_points("a", pts)
    assert st.spectrums["a"].points_completed is True
    assert st.version.get(spectrum_version_key("a")) == v0 + 1


def test_set_points_empty_completes_stage_without_available_points():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    v0 = st.version.get(spectrum_version_key("a"))
    empty = PointsData(
        dev_values=np.array([], dtype=np.float64),
        fluxs=np.array([], dtype=np.float64),
        freqs=np.array([], dtype=np.float64),
    )
    st.set_points("a", empty)
    assert st.spectrums["a"].points_completed is True
    assert st.spectrums["a"].point_count == 0
    assert st.version.get(spectrum_version_key("a")) == v0 + 1


@pytest.mark.parametrize(
    "invalid", [np.array([1.0]), np.array([[1.0, 2.0]]), np.array([np.nan, 2.0])]
)
def test_alignment_rejects_invalid_point_mapping_without_mutation(invalid):
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    st.set_points(
        "a",
        PointsData(
            dev_values=np.array([0.0, 1.0]),
            fluxs=np.array([0.5, 1.0]),
            freqs=np.array([5.0, 5.1]),
        ),
    )
    entry = st.spectrums["a"]
    version = st.version.get(spectrum_version_key("a"))
    raw_fluxs = entry.raw["fluxs"].copy()
    with pytest.raises(ValueError, match="new_point_fluxs|finite"):
        st.set_alignment(
            "a",
            AlignmentPublication(0.0, 1.0, 2.0),
            np.array([0.5, 0.75, 1.0]),
            invalid,
        )
    assert st.spectrums["a"] is entry
    np.testing.assert_array_equal(entry.raw["fluxs"], raw_fluxs)
    np.testing.assert_array_equal(entry.points["fluxs"], [0.5, 1.0])
    assert st.version.get(spectrum_version_key("a")) == version


def test_set_selection_bumps_selection():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    st.set_points(
        "a",
        PointsData(
            dev_values=np.arange(3.0), fluxs=np.arange(3.0), freqs=np.arange(3.0)
        ),
    )
    st.set_selection(np.array([True, False, True]))
    assert st.version.get(SELECTION_VERSION_KEY) == 1
    assert st.selection_mask() is not None


def _three_point_state() -> FluxDepState:
    st = FluxDepState()
    for name in ("a", "b"):
        st.put_spectrum(_make_entry(name))
        st.set_alignment(
            name,
            AlignmentPublication(0.0, 1.0, 2.0, None),
            np.linspace(0.0, 1.0, 3),
            np.empty(0),
        )
        three = np.arange(3.0)
        st.set_points(name, PointsData(dev_values=three, fluxs=three, freqs=three))
    return st


def test_brush_mask_projects_the_published_brush_beside_the_kept_mask():
    st = _three_point_state()
    assert st.brush_mask() is None
    kept = np.array([True, False, False, False, True, False])
    brush = np.array([True, True, False, False, True, True])
    st.set_selection(kept, 0.1, brush)
    np.testing.assert_array_equal(st.selection_mask(), kept)
    np.testing.assert_array_equal(st.brush_mask(), brush)
    # Without a separate brush the kept mask is the brush.
    st.set_selection(kept, 0.0)
    np.testing.assert_array_equal(st.brush_mask(), kept)


@pytest.mark.parametrize(
    ("brush", "message"),
    [
        (np.ones(5, dtype=np.bool_), "brush mask must match"),
        (np.array([False, True, True, True, True, True]), "kept point"),
    ],
)
def test_set_selection_rejects_a_brush_that_is_misshaped_or_misses_kept_points(
    brush, message: str
):
    st = _three_point_state()
    before = st.version.snapshot()
    with pytest.raises(ValueError, match=message):
        st.set_selection(np.array([True, False, False, False, True, False]), 0.1, brush)
    assert st.brush_mask() is None
    assert st.selection_mask() is None
    assert st.version.snapshot() == before


@pytest.mark.parametrize("change", ["replace", "remove", "set_points", "reset_points"])
def test_paths_that_drop_a_filter_mask_drop_its_brush(change: str):
    st = _three_point_state()
    st.set_selection(
        np.array([True, False, False, False, True, False]),
        0.1,
        np.array([True, True, False, False, True, True]),
    )
    if change == "replace":
        st.put_spectrum(_make_entry("b"))
    elif change == "remove":
        st.remove_spectrum("b")
    elif change == "set_points":
        st.set_points(
            "b",
            PointsData(
                dev_values=np.arange(2.0), fluxs=np.arange(2.0), freqs=np.arange(2.0)
            ),
        )
    else:
        st.reset_points("b")
    entry_b = st.spectrums.get("b")
    assert entry_b is None or (
        entry_b.filter_mask is None and entry_b.brush_mask is None
    )
    brush = st.brush_mask()
    assert brush is not None
    np.testing.assert_array_equal(brush[:3], [True, True, False])
    assert brush[3:].all()


def test_set_project_bumps_project():
    st = FluxDepState()
    st.set_project(None)
    assert st.version.get(PROJECT_VERSION_KEY) == 1
    assert st.project is None


def test_set_fit_params_bumps_and_clears_result():
    from zcu_tools.analysis.fluxdep.models import TransitionDict
    from zcu_tools.gui.app.fluxdep.state import FIT_VERSION_KEY

    st = FluxDepState()
    st.set_fit_result((5.0, 1.0, 0.5))  # a stale result
    assert st.fit.has_result
    before = st.version.get(FIT_VERSION_KEY)
    st.set_fit_params(
        "db.h5",
        ParamBounds(EJ=(2.0, 7.0), EC=(0.5, 1.5), EL=(0.2, 0.8)),
        TransitionDict({"transitions": [(0, 1)]}),
        5.5,
        9.0,
    )
    # changing inputs clears the stale result and bumps the version
    assert st.fit.params is None
    assert st.fit.search_db == "db.h5"
    assert st.fit.r_f == 5.5
    assert st.version.get(FIT_VERSION_KEY) == before + 1


def test_set_fit_result_records_params():
    from zcu_tools.gui.app.fluxdep.state import FIT_VERSION_KEY

    st = FluxDepState()
    before = st.version.get(FIT_VERSION_KEY)
    st.set_fit_result((4.2, 1.1, 0.3))
    assert st.fit.params == (4.2, 1.1, 0.3)
    assert st.fit.has_result
    assert st.version.get(FIT_VERSION_KEY) == before + 1


# --- transitions r_f/sample_f helpers (Optional) ---------------------------


def test_transitions_need_r_f():
    from zcu_tools.analysis.fluxdep.models import TransitionDict
    from zcu_tools.gui.app.fluxdep.state import transitions_need_r_f

    assert transitions_need_r_f(TransitionDict({"red side": [(0, 1)]}))
    assert transitions_need_r_f(TransitionDict({"blue side": [(0, 1)]}))
    assert not transitions_need_r_f(TransitionDict({"transitions": [(0, 1)]}))
    assert not transitions_need_r_f(
        TransitionDict({"mirror": [(0, 1)]})
    )  # needs sample_f


def test_transitions_need_sample_f():
    from zcu_tools.analysis.fluxdep.models import TransitionDict
    from zcu_tools.gui.app.fluxdep.state import transitions_need_sample_f

    assert transitions_need_sample_f(TransitionDict({"mirror": [(0, 1)]}))
    assert transitions_need_sample_f(TransitionDict({"mirror blue": [(0, 1)]}))
    assert not transitions_need_sample_f(TransitionDict({"transitions": [(0, 1)]}))


def test_transitions_with_freqs_injects_only_set():
    from zcu_tools.analysis.fluxdep.models import TransitionDict
    from zcu_tools.gui.app.fluxdep.state import transitions_with_freqs

    base = TransitionDict({"mirror": [(0, 1)]})
    # only sample_f set → only sample_f key added; r_f stays absent
    out = transitions_with_freqs(base, None, 9.5)
    assert out.get("sample_f") == 9.5
    assert "r_f" not in out
    # both set
    out2 = transitions_with_freqs(base, 5.5, 9.5)
    assert out2.get("r_f") == 5.5 and out2.get("sample_f") == 9.5
    # original not mutated
    assert "r_f" not in base and "sample_f" not in base


def test_fit_state_freqs_default_none():
    st = FluxDepState()
    assert st.fit.r_f is None
    assert st.fit.sample_f is None
