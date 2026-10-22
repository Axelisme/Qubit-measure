"""Tests for dispersive-fit-gui state: ProjectInfo derivation + version discipline."""

from __future__ import annotations

import numpy as np
from zcu_tools.analysis.dispersive.models import PreprocessResult
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.dispersive.state import (
    DEFAULT_BARE_RF,
    FIT_VERSION_KEY,
    ONETONE_VERSION_KEY,
    PREPROCESS_VERSION_KEY,
    PROJECT_VERSION_KEY,
    DispersiveState,
    FluxoniumInputs,
    OnetoneEntry,
    VersionTable,
)


def _make_onetone(name: str = "r1") -> OnetoneEntry:
    e = np.linspace(0.0, 1.0, 3).astype(np.float64)
    raw = SpectrumData(
        dev_values=e.copy(),
        fluxs=e.copy(),
        freqs=e.copy(),
        signals=np.zeros((3, 3), dtype=np.complex128),
    )
    return OnetoneEntry(name=name, raw=raw)


def _make_preprocess(signature: tuple[object, ...] = ("a",)) -> PreprocessResult:
    e = np.linspace(0.0, 1.0, 3).astype(np.float64)
    return PreprocessResult(
        sp_fluxs=e.copy(),
        sp_freqs=e.copy(),
        norm_phases=np.zeros((3, 3), dtype=np.float64),
        edelays=e.copy(),
        edelay=0.0,
        signature=signature,
    )


def _inputs(bare_rf: float = 5.3) -> FluxoniumInputs:
    return FluxoniumInputs(
        params=(4.0, 1.0, 0.5),
        flux_half=0.5,
        flux_bias=0.0,
        flux_period=1.0,
        bare_rf_seed=bare_rf,
    )


# --- VersionTable is the shared mechanism ------------------------------


def test_version_table_is_shared_mechanism():
    from zcu_tools.gui.version_table import VersionTable as Shared

    assert VersionTable is Shared


# --- setter version discipline -----------------------------------------


def test_set_project_bumps_project_version(analysis_target):
    st = DispersiveState()
    before = st.version.get(PROJECT_VERSION_KEY)
    st.set_project(analysis_target)
    assert st.version.get(PROJECT_VERSION_KEY) > before


def test_set_fit_inputs_seeds_bare_rf_and_bumps_project():
    st = DispersiveState()
    before = st.version.get(PROJECT_VERSION_KEY)
    st.set_fit_inputs(_inputs(bare_rf=5.3))
    assert st.fit_inputs is not None
    assert st.disp_fit.bare_rf == 5.3  # seeded since it had no value
    assert st.version.get(PROJECT_VERSION_KEY) > before


def test_set_fit_inputs_does_not_clobber_existing_bare_rf():
    st = DispersiveState()
    st.set_disp_result(g=0.06, bare_rf=5.9, res_dim=4)
    st.set_fit_inputs(_inputs(bare_rf=5.3))
    assert st.disp_fit.bare_rf == 5.9  # kept the prior value


def test_set_onetone_drops_preprocess_and_fit():
    st = DispersiveState()
    st.set_preprocess(_make_preprocess())
    st.set_disp_result(g=0.06, bare_rf=5.3, res_dim=4)
    pre_pp = st.version.get(PREPROCESS_VERSION_KEY)
    pre_fit = st.version.get(FIT_VERSION_KEY)

    st.set_onetone(_make_onetone())

    assert st.preprocess is None
    assert st.disp_fit.g is None  # fit cleared
    assert st.version.get(ONETONE_VERSION_KEY) > 0
    # dropped keys reset to 0 (drop_prefix removes them).
    assert st.version.get(PREPROCESS_VERSION_KEY) == 0 < pre_pp
    assert st.version.get(FIT_VERSION_KEY) == 0 < pre_fit


def test_set_preprocess_invalidates_fit_on_signature_change():
    st = DispersiveState()
    st.set_preprocess(_make_preprocess(signature=("a",)))
    st.set_disp_result(g=0.06, bare_rf=5.3, res_dim=4)
    # re-preprocess with a DIFFERENT signature → fit goes stale
    st.set_preprocess(_make_preprocess(signature=("b",)))
    assert st.disp_fit.g is None


def test_set_preprocess_keeps_fit_on_same_signature():
    st = DispersiveState()
    st.set_preprocess(_make_preprocess(signature=("a",)))
    st.set_disp_result(g=0.06, bare_rf=5.3, res_dim=4)
    st.set_preprocess(_make_preprocess(signature=("a",)))
    assert st.disp_fit.g == 0.06  # unchanged


def test_set_disp_result_records_and_bumps_fit():
    st = DispersiveState()
    before = st.version.get(FIT_VERSION_KEY)
    st.set_disp_result(g=0.07, bare_rf=5.35, res_dim=5)
    assert st.disp_fit.has_result is True
    assert st.disp_fit.g == 0.07 and st.disp_fit.bare_rf == 5.35
    assert st.disp_fit.res_dim == 5
    assert st.version.get(FIT_VERSION_KEY) > before


def test_default_bare_rf_constant():
    assert DEFAULT_BARE_RF == 5.0
