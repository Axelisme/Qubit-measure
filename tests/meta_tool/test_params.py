from __future__ import annotations

import json
from datetime import datetime

import pytest
from zcu_tools.meta_tool import (
    DispersiveFit,
    FluxDepFit,
    ParamsProject,
    QubitParams,
    QubitParamsError,
    T1CurveFit,
    T1CurveFitParams,
    T1CurveFitUncertainty,
    params_path_for_result_dir,
)


def _fit(*, r_f: float | None = 5.3) -> FluxDepFit:
    transitions = {} if r_f is None else {"r_f": r_f, "transitions": [(0, 1)]}
    return FluxDepFit(
        EJ=4.0,
        EC=1.0,
        EL=0.5,
        flux_half=0.5,
        flux_int=1.0,
        flux_period=2.0,
        plot_transitions=transitions,
    )


def _assert_timestamp(value: object) -> str:
    assert isinstance(value, str)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    return value


def test_ensure_project_creates_canonical_identity(tmp_path) -> None:
    path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"

    QubitParams(path).ensure_project(ParamsProject("ChipA", "Q1", "R1"))

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["project"] == {
        "chip_name": "ChipA",
        "qubit_name": "Q1",
        "resonator_name": "R1",
    }
    assert raw["name"] == "ChipA/Q1"


def test_migrate_project_from_path_preserves_sections(tmp_path) -> None:
    path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"name": "stale/wrong", "fluxdep_fit": {"params": {}}}),
        encoding="utf-8",
    )

    project = QubitParams(path).migrate_project_from_path(
        result_root=tmp_path / "result"
    )

    assert project == ParamsProject("ChipA", "Q1")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["project"]["chip_name"] == "ChipA"
    assert raw["project"]["qubit_name"] == "Q1"
    assert raw["fluxdep_fit"] == {"params": {}}


def test_migrate_project_from_single_level_path_uses_same_chip_and_qubit(
    tmp_path,
) -> None:
    path = tmp_path / "result" / "Si001" / "params.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"name": "legacy"}), encoding="utf-8")

    project = QubitParams(path).migrate_project_from_path(
        result_root=tmp_path / "result"
    )

    assert project == ParamsProject("Si001", "Si001")


def test_set_fluxdep_fit_preserves_independent_dispersive_and_stamps_section(
    tmp_path,
) -> None:
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "name": "ChipA/Q1",
                "dispersive": {"g": 0.05, "bare_rf": 5.8},
                "extra": {"keep": True},
            }
        ),
        encoding="utf-8",
    )

    QubitParams(path).set_fluxdep_fit(_fit())

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["dispersive"] == {"g": 0.05, "bare_rf": 5.8}
    assert raw["extra"] == {"keep": True}
    assert raw["fluxdep_fit"]["params"] == {"EJ": 4.0, "EC": 1.0, "EL": 0.5}
    timestamp = _assert_timestamp(raw["fluxdep_fit"]["timestamp"])
    assert QubitParams(path).require_fluxdep_fit().timestamp == timestamp


def test_set_dispersive_fit_preserves_fluxdep_fit(tmp_path) -> None:
    path = tmp_path / "params.json"
    params = QubitParams(path)
    params.ensure_project(ParamsProject("ChipA", "Q1"))
    params.set_fluxdep_fit(_fit())

    params.set_dispersive_fit(DispersiveFit(g=0.068, bare_rf=5.35))

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["dispersive"]["g"] == 0.068
    assert raw["dispersive"]["bare_rf"] == 5.35
    timestamp = _assert_timestamp(raw["dispersive"]["timestamp"])
    assert QubitParams(path).require_fluxdep_fit().timestamp is not None
    assert QubitParams(path).get_dispersive_fit() == DispersiveFit(
        g=0.068,
        bare_rf=5.35,
        timestamp=timestamp,
    )
    assert raw["fluxdep_fit"]["flux_period"] == 2.0


def test_set_dispersive_fit_requires_existing_fluxdep_fit(tmp_path) -> None:
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"name": "ChipA/Q1"}), encoding="utf-8")

    with pytest.raises(QubitParamsError, match="fluxdep_fit"):
        QubitParams(path).set_dispersive_fit(DispersiveFit(g=0.068, bare_rf=5.35))


def test_require_dispersive_inputs_prefers_dispersive_bare_rf(tmp_path) -> None:
    params = QubitParams(tmp_path / "params.json")
    params.set_fluxdep_fit(_fit(r_f=5.3))
    params.set_dispersive_fit(DispersiveFit(g=0.068, bare_rf=5.9))

    inputs = params.require_dispersive_inputs(default_bare_rf=5.0)

    assert inputs.params == (4.0, 1.0, 0.5)
    assert inputs.flux_half == 0.5
    assert inputs.flux_int == 1.0
    assert inputs.flux_period == 2.0
    assert inputs.bare_rf_seed == 5.9


def test_require_dispersive_inputs_falls_back_to_rf_then_default(tmp_path) -> None:
    with_rf = QubitParams(tmp_path / "with_rf.json")
    with_rf.set_fluxdep_fit(_fit(r_f=5.3))
    assert with_rf.require_dispersive_inputs(default_bare_rf=5.0).bare_rf_seed == 5.3

    without_rf = QubitParams(tmp_path / "without_rf.json")
    without_rf.set_fluxdep_fit(_fit(r_f=None))
    assert without_rf.require_dispersive_inputs(default_bare_rf=5.0).bare_rf_seed == 5.0


def test_require_fluxonium_model_reads_predictor_fields(tmp_path) -> None:
    params = QubitParams(tmp_path / "params.json")
    params.set_fluxdep_fit(_fit())

    model = params.require_fluxonium_model(flux_bias=0.12)

    assert model.params == (4.0, 1.0, 0.5)
    assert model.flux_half == 0.5
    assert model.flux_period == 2.0
    assert model.flux_bias == 0.12


def test_set_t1_curve_fit_preserves_independent_sections_and_round_trips(
    tmp_path,
) -> None:
    path = tmp_path / "params.json"
    params = QubitParams(path)
    params.ensure_project(ParamsProject("ChipA", "Q1"))
    params.set_fluxdep_fit(_fit())
    params.set_dispersive_fit(DispersiveFit(g=0.068, bare_rf=5.35))

    params.set_t1_curve_fit(
        T1CurveFit(
            params=T1CurveFitParams(
                Temp=0.055,
                Q_cap=7.2e5,
                x_qp=1.8e-6,
                Q_ind=2.4e7,
            ),
            stderr=T1CurveFitUncertainty(
                Q_cap=1.1e4,
                x_qp=2.0e-8,
                Q_ind=float("inf"),
                Temp=None,
            ),
            fixed=("Q_ind",),
            free=("Q_cap", "x_qp", "Temp"),
            cost=1.25,
            reduced_chi2=0.42,
            success=True,
            message="converged",
            residual_mode="log",
            loss="soft_l1",
            max_nfev=200,
            init=T1CurveFitParams(
                Temp=0.08,
                Q_cap=5.0e5,
                x_qp=2.5e-6,
                Q_ind=2.4e7,
            ),
            bounds={
                "Q_cap": (1.0e4, 1.0e8),
                "x_qp": (1.0e-9, 1.0e-3),
                "Q_ind": (1.0e5, 1.0e10),
                "Temp": (10e-3, 300e-3),
            },
        )
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    timestamp = _assert_timestamp(raw["t1_curve_fit"]["timestamp"])
    assert raw["fluxdep_fit"]["flux_period"] == 2.0
    assert raw["dispersive"]["bare_rf"] == 5.35
    assert raw["t1_curve_fit"]["stderr"]["Q_ind"] is None
    assert raw["t1_curve_fit"]["stderr"]["Temp"] is None

    assert QubitParams(path).require_t1_curve_fit() == T1CurveFit(
        params=T1CurveFitParams(
            Temp=0.055,
            Q_cap=7.2e5,
            x_qp=1.8e-6,
            Q_ind=2.4e7,
        ),
        stderr=T1CurveFitUncertainty(
            Q_cap=1.1e4,
            x_qp=2.0e-8,
            Q_ind=None,
            Temp=None,
        ),
        fixed=("Q_ind",),
        free=("Q_cap", "x_qp", "Temp"),
        cost=1.25,
        reduced_chi2=0.42,
        success=True,
        message="converged",
        residual_mode="log",
        loss="soft_l1",
        max_nfev=200,
        init=T1CurveFitParams(
            Temp=0.08,
            Q_cap=5.0e5,
            x_qp=2.5e-6,
            Q_ind=2.4e7,
        ),
        bounds={
            "Q_cap": (1.0e4, 1.0e8),
            "x_qp": (1.0e-9, 1.0e-3),
            "Q_ind": (1.0e5, 1.0e10),
            "Temp": (10e-3, 300e-3),
        },
        timestamp=timestamp,
    )


def test_set_t1_curve_fit_round_trips_partial_noise_whitelist(tmp_path) -> None:
    path = tmp_path / "params.json"
    params = QubitParams(path)
    params.ensure_project(ParamsProject("ChipA", "Q1"))
    params.set_fluxdep_fit(_fit())

    params.set_t1_curve_fit(
        T1CurveFit(
            params=T1CurveFitParams(
                Temp=0.055,
                Q_cap=7.2e5,
                x_qp=1.8e-6,
            ),
            stderr=T1CurveFitUncertainty(
                Q_cap=1.1e4,
                x_qp=2.0e-8,
                Temp=0.0,
            ),
            fixed=("Temp",),
            free=("Q_cap", "x_qp"),
            cost=1.25,
            reduced_chi2=0.42,
            success=True,
            message="converged",
            residual_mode="log",
            loss="soft_l1",
            max_nfev=200,
            init=T1CurveFitParams(
                Temp=0.08,
                Q_cap=5.0e5,
                x_qp=2.5e-6,
            ),
            bounds={
                "Q_cap": (1.0e4, 1.0e8),
                "x_qp": (1.0e-9, 1.0e-3),
                "Temp": (10e-3, 300e-3),
            },
        )
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    timestamp = _assert_timestamp(raw["t1_curve_fit"]["timestamp"])
    section = raw["t1_curve_fit"]
    assert "Q_ind" not in section["params"]
    assert "Q_ind" not in section["stderr"]
    assert "Q_ind" not in section["init"]
    assert "Q_ind" not in section["bounds"]

    assert QubitParams(path).require_t1_curve_fit() == T1CurveFit(
        params=T1CurveFitParams(
            Temp=0.055,
            Q_cap=7.2e5,
            x_qp=1.8e-6,
        ),
        stderr=T1CurveFitUncertainty(
            Q_cap=1.1e4,
            x_qp=2.0e-8,
            Temp=0.0,
        ),
        fixed=("Temp",),
        free=("Q_cap", "x_qp"),
        cost=1.25,
        reduced_chi2=0.42,
        success=True,
        message="converged",
        residual_mode="log",
        loss="soft_l1",
        max_nfev=200,
        init=T1CurveFitParams(
            Temp=0.08,
            Q_cap=5.0e5,
            x_qp=2.5e-6,
        ),
        bounds={
            "Q_cap": (1.0e4, 1.0e8),
            "x_qp": (1.0e-9, 1.0e-3),
            "Temp": (10e-3, 300e-3),
        },
        timestamp=timestamp,
    )


def test_t1_curve_fit_missing_and_orphan_writes_fast_fail(tmp_path) -> None:
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"name": "ChipA/Q1"}), encoding="utf-8")
    fit = T1CurveFit(
        params=T1CurveFitParams(
            Temp=0.055,
            Q_cap=7.2e5,
            x_qp=1.8e-6,
            Q_ind=2.4e7,
        )
    )

    with pytest.raises(QubitParamsError, match="t1_curve_fit"):
        QubitParams(path).require_t1_curve_fit()
    with pytest.raises(QubitParamsError, match="fluxdep_fit"):
        QubitParams(path).set_t1_curve_fit(fit)


def test_t1_curve_fit_rejects_unknown_bound_on_write(tmp_path) -> None:
    path = tmp_path / "params.json"
    params = QubitParams(path)
    params.ensure_project(ParamsProject("ChipA", "Q1"))
    params.set_fluxdep_fit(_fit())

    with pytest.raises(QubitParamsError, match="known T1 fit parameter"):
        params.set_t1_curve_fit(
            T1CurveFit(
                params=T1CurveFitParams(
                    Temp=0.055,
                    Q_cap=7.2e5,
                    x_qp=1.8e-6,
                    Q_ind=2.4e7,
                ),
                bounds={"bad": (1.0, 2.0)},
            )
        )


def test_t1_curve_fit_rejects_inactive_metadata_on_write(tmp_path) -> None:
    path = tmp_path / "params.json"
    params = QubitParams(path)
    params.ensure_project(ParamsProject("ChipA", "Q1"))
    params.set_fluxdep_fit(_fit())

    with pytest.raises(QubitParamsError, match="inactive"):
        params.set_t1_curve_fit(
            T1CurveFit(
                params=T1CurveFitParams(
                    Temp=0.055,
                    Q_cap=7.2e5,
                    x_qp=1.8e-6,
                ),
                fixed=("Q_ind",),
            )
        )


def test_missing_file_fast_fails_on_read(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        QubitParams(tmp_path / "missing.json").require_fluxdep_fit()


def test_params_path_for_result_dir() -> None:
    assert params_path_for_result_dir("result/ChipA/Q1").endswith(
        "result/ChipA/Q1/params.json"
    )
