"""Predictor handoff through the public dispersive project service."""

import json

import pytest
from zcu_tools.gui.app.dispersive.services.project import ProjectService
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.resources.analysis_source import AnalysisSource
from zcu_tools.resources.dispersive_file import (
    DispersiveFile,
    DispersiveParams,
    PlotTransitions,
    save_dispersive_file,
)
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorFlux,
    PredictorSource,
    save_predictor_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams


@pytest.mark.parametrize(
    ("bare_rf", "r_f", "resonator", "expected"),
    [
        (5.9, 5.4, True, 5.9),
        (None, 5.4, True, 5.4),
        (None, None, True, 5.75),
        (None, None, False, 5.0),
    ],
)
def test_bare_rf_seed_priority(analysis_target, bare_rf, r_f, resonator, expected):
    resource = analysis_target.resource
    if not resonator:
        resource.save_aliases(EntryAliases(aliases={"qubit": "q1"}))
    save_dispersive_file(
        resource.dispersive_path / "q1.json",
        DispersiveFile(
            format="zcu.dispersive",
            version=1,
            params=DispersiveParams(
                bare_rf=bare_rf, plot_transitions=PlotTransitions(r_f=r_f)
            ),
            source=AnalysisSource(run_id=None, created_at=None),
        ),
    )
    inputs = ProjectService(DispersiveState(analysis_target)).load_fit_inputs(
        "fit.json"
    )
    assert inputs.bare_rf_seed == expected


def test_predictor_reload_preserves_accepted_tune(analysis_target):
    state = DispersiveState(analysis_target)
    state.set_disp_result(g=0.06, bare_rf=5.91, res_dim=4)
    ProjectService(state).load_fit_inputs("fit.json")
    assert state.disp_fit.bare_rf == 5.91


def test_load_predictor_reads_energies_and_recorded_flux(analysis_target):
    state = DispersiveState(analysis_target)
    inputs = ProjectService(state).load_fit_inputs("fit.json")
    assert inputs.params == (4.0, 1.0, 0.5)
    assert inputs.flux.to_dict() == {
        "offset": 0.625,
        "period": {"bias": -2.0},
        "unit": {"bias": "V"},
    }


def test_predictor_listing_is_entry_local(analysis_target, tmp_path):
    (analysis_target.resource.predictors_path / "notes.txt").write_text("notes")
    (tmp_path / "outside.json").write_text("{}")
    assert ProjectService(DispersiveState(analysis_target)).list_predictors() == (
        "fit.json",
    )


@pytest.mark.parametrize("filename", ["../outside.json", "/outside.json"])
def test_predictor_selection_rejects_nonlocal_names(analysis_target, filename):
    with pytest.raises(ValueError, match="predictor"):
        ProjectService(DispersiveState(analysis_target)).load_fit_inputs(filename)


def test_incomplete_predictor_rejected_without_replacing_inputs(analysis_target):
    state = DispersiveState(analysis_target)
    service = ProjectService(state)
    original = service.load_fit_inputs("fit.json")
    save_predictor_file(
        analysis_target.resource.predictors_path / "partial.json",
        PredictorFile(
            model="fluxonium",
            params=FluxoniumParams(EJ=7.0),
            flux=PredictorFlux(0.625, {"bias": -2.0}, {"bias": "V"}),
            source=PredictorSource(run_id=None, created_at=None),
        ),
    )
    with pytest.raises(ValueError, match="Missing parameters"):
        service.load_fit_inputs("partial.json")
    # A failed read must not publish a partial parameter set.
    assert state.fit_inputs is original


def test_missing_predictor_flux_does_not_replace_inputs(analysis_target):
    state = DispersiveState(analysis_target)
    service = ProjectService(state)
    original = service.load_fit_inputs("fit.json")
    path = analysis_target.resource.predictors_path / "missing.json"
    path.write_text(
        json.dumps(
            {
                "format": "zcu.predictor",
                "version": 2,
                "model": "fluxonium",
                "params": {"EJ": 4.0, "EC": 1.0, "EL": 0.5},
                "source": {"run_id": None, "created_at": None},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing fields.*flux"):
        service.load_fit_inputs("missing.json")
    assert state.fit_inputs is original
