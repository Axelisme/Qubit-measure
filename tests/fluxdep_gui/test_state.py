"""Tests for fluxdep-gui state: VersionTable (copied mechanism) + FluxDepState."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.state import (
    PROJECT_VERSION_KEY,
    SELECTION_VERSION_KEY,
    SPECTRUM_SET_VERSION_KEY,
    FluxDepState,
    ProjectInfo,
    SpectrumEntry,
    SpecType,
    VersionTable,
    spectrum_version_key,
)
from zcu_tools.notebook.persistance import PointsData, SpectrumData


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
# VersionTable (copied verbatim from measure — verify mechanism intact)
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


def test_remove_spectrum_drops_key_bumps_set():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    st.remove_spectrum("a")
    assert "a" not in st.spectrums
    assert st.version.get(spectrum_version_key("a")) == 0  # dropped
    assert st.version.get(SPECTRUM_SET_VERSION_KEY) == 2  # add + remove


def test_remove_active_spectrum_clears_active():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    st.set_active("a")
    st.remove_spectrum("a")
    assert st.active_spectrum is None


def test_set_active_unknown_raises():
    st = FluxDepState()
    with pytest.raises(KeyError):
        st.set_active("nope")


def test_set_alignment_marks_aligned_and_bumps():
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    v0 = st.version.get(spectrum_version_key("a"))
    new_fluxs = st.spectrums["a"].raw["dev_values"].copy()
    st.set_alignment(
        "a", flux_half=1.0, flux_int=2.0, flux_period=2.0, new_fluxs=new_fluxs
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
    assert st.spectrums["a"].points_selected is True
    assert st.version.get(spectrum_version_key("a")) == v0 + 1


def test_set_points_empty_does_not_mark_selected():
    # an empty point set (user deselected everything) is recorded but must not be
    # flagged points_selected — downstream readers would crash on an empty cloud.
    st = FluxDepState()
    st.put_spectrum(_make_entry("a"))
    v0 = st.version.get(spectrum_version_key("a"))
    empty = PointsData(
        dev_values=np.array([], dtype=np.float64),
        fluxs=np.array([], dtype=np.float64),
        freqs=np.array([], dtype=np.float64),
    )
    st.set_points("a", empty)
    assert st.spectrums["a"].points_selected is False
    assert st.version.get(spectrum_version_key("a")) == v0 + 1


def test_set_selection_bumps_selection():
    st = FluxDepState()
    st.set_selection(np.array([True, False, True]))
    assert st.version.get(SELECTION_VERSION_KEY) == 1
    assert st.selection.selected is not None


def test_set_project_bumps_project():
    st = FluxDepState()
    st.set_project(ProjectInfo(chip_name="Q5_2D", qub_name="Q1"))
    assert st.version.get(PROJECT_VERSION_KEY) == 1
    assert st.project.chip_name == "Q5_2D"


def test_project_info_derives_paths_from_names():
    import os

    # unset result_dir / database_path are derived from chip/qubit in __post_init__
    p = ProjectInfo(chip_name="Q5_2D", qub_name="Q1")
    assert p.result_dir == os.path.join("result", "Q5_2D", "Q1")
    # raw spectra live under Database/, not result/ (cf. measure-gui save layout)
    assert p.database_path == os.path.join("Database", "Q5_2D", "Q1")


def test_project_info_default_names_derive_unknown_paths():
    import os

    # a bare ProjectInfo still has concrete (never empty) paths
    p = ProjectInfo()
    assert p.result_dir == os.path.join("result", "unknown_chip", "unknown_qubit")
    assert p.database_path == os.path.join("Database", "unknown_chip", "unknown_qubit")


def test_project_info_explicit_paths_override_derivation():
    # a provided path is kept as the user's override, not re-derived
    p = ProjectInfo(
        chip_name="Q5_2D",
        qub_name="Q1",
        result_dir="/custom/out",
        database_path="/custom/raw",
    )
    assert p.result_dir == "/custom/out"
    assert p.database_path == "/custom/raw"


def test_project_info_root_dir_anchors_derived_defaults():
    import os

    # root_dir (the repo root, injected by the entry script) anchors the derived
    # defaults there instead of leaving them relative to cwd — the .bat-launcher
    # fix: a launcher that cd's into script/ must not scope defaults under script/.
    root = os.path.join(os.sep, "repo")
    p = ProjectInfo(chip_name="Q5_2D", qub_name="Q1", root_dir=root)
    assert p.result_dir == os.path.join(root, "result", "Q5_2D", "Q1")
    assert p.database_path == os.path.join(root, "Database", "Q5_2D", "Q1")


def test_project_info_empty_root_dir_keeps_relative_default():
    import os

    # No injection (tests / direct runs from the repo root) → the legacy
    # cwd-relative default, so nothing regresses.
    p = ProjectInfo(chip_name="Q5_2D", qub_name="Q1", root_dir="")
    assert p.result_dir == os.path.join("result", "Q5_2D", "Q1")


# --- FitState (v2) ---------------------------------------------------------


def test_set_fit_params_bumps_and_clears_result():
    from zcu_tools.gui.app.fluxdep.state import FIT_VERSION_KEY
    from zcu_tools.notebook.persistance import TransitionDict

    st = FluxDepState()
    st.set_fit_result((5.0, 1.0, 0.5))  # a stale result
    assert st.fit.has_result
    before = st.version.get(FIT_VERSION_KEY)
    st.set_fit_params(
        "db.h5",
        (2.0, 7.0),
        (0.5, 1.5),
        (0.2, 0.8),
        TransitionDict({"transitions": [(0, 1)]}),
        5.5,
        9.0,
    )
    # changing inputs clears the stale result and bumps the version
    assert st.fit.params is None
    assert st.fit.database_path == "db.h5"
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


def test_default_transitions_is_basic_preset():
    from zcu_tools.gui.app.fluxdep.state import default_transitions

    t = default_transitions()
    assert t["transitions"] == [(0, 1), (0, 2), (1, 2), (1, 3)]


# --- transitions r_f/sample_f helpers (Optional) ---------------------------


def test_transitions_need_r_f():
    from zcu_tools.gui.app.fluxdep.state import transitions_need_r_f
    from zcu_tools.notebook.persistance import TransitionDict

    assert transitions_need_r_f(TransitionDict({"red side": [(0, 1)]}))
    assert transitions_need_r_f(TransitionDict({"blue side": [(0, 1)]}))
    assert not transitions_need_r_f(TransitionDict({"transitions": [(0, 1)]}))
    assert not transitions_need_r_f(
        TransitionDict({"mirror": [(0, 1)]})
    )  # needs sample_f


def test_transitions_need_sample_f():
    from zcu_tools.gui.app.fluxdep.state import transitions_need_sample_f
    from zcu_tools.notebook.persistance import TransitionDict

    assert transitions_need_sample_f(TransitionDict({"mirror": [(0, 1)]}))
    assert transitions_need_sample_f(TransitionDict({"mirror blue": [(0, 1)]}))
    assert not transitions_need_sample_f(TransitionDict({"transitions": [(0, 1)]}))


def test_transitions_with_freqs_injects_only_set():
    from zcu_tools.gui.app.fluxdep.state import transitions_with_freqs
    from zcu_tools.notebook.persistance import TransitionDict

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
