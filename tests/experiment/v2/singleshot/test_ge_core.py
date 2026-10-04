"""GE's numeric FIT/post and canonical Result through its public interface."""

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import calc_population_pdf, compute_fit_quality
from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.utils.single_shot.ge import base as ge_fitting
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEModuleCfg,
    GEPostAnalyzeOptions,
)
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def ge_source() -> RunRecord[GE_Cfg, GE_Result]:
    rng = np.random.default_rng(83)
    excited = rng.random((2, 6000)) < np.array([0.1, 0.9])[:, None]
    signals = np.asarray(
        np.where(excited, 1 + 0.4j, -1 - 0.4j)
        + 0.2 * (rng.normal(size=excited.shape) + 1j * rng.normal(size=excited.shape)),
        dtype=np.complex128,
    )
    cfg = GE_Cfg(
        reps=1,
        rounds=1,
        shots=6000,
        modules=GEModuleCfg(
            probe_pulse=PulseCfg(
                ch=0,
                nqz=1,
                gain=0.5,
                freq=4000.0,
                phase=0.0,
                waveform=ConstWaveformCfg(length=1.0),
            ),
            readout=PulseReadoutCfg(
                pulse_cfg=PulseCfg(
                    ch=0,
                    nqz=1,
                    gain=0.2,
                    freq=7000.0,
                    phase=0.0,
                    waveform=ConstWaveformCfg(length=1.0),
                ),
                ro_cfg=DirectReadoutCfg(
                    ro_ch=0, ro_length=1.0, ro_freq=7000.0, trig_offset=0.0
                ),
            ),
        ),
    )
    return RunRecord(cfg, GE_Result(signals, np.arange(6000), np.array([0, 1])))


def test_quality_keeps_joint_and_population_fit_inputs_and_covariances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joint_parameters = (-1.0, 1.0, 0.2, 0.8, 0.1, 0.5, 0.01)
    joint_covariance = np.diag([0.01, 0.04, 0.0016, 0.0004, 0.0001, 0.0025, 0.0001])
    populations = [(0.85, 0.1, 0.01), (0.15, 0.8, 0.02)]
    covariances = [
        np.diag([0.007225, 0.0004, 0.000004]),
        np.diag([0.0036, 0.0144, 0.000016]),
    ]
    joint_inputs: list[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    population_inputs: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    def joint(
        xs: NDArray[np.float64],
        ground: NDArray[np.float64],
        excited: NDArray[np.float64],
        **kwargs: object,
    ):
        joint_inputs.append((xs.copy(), ground.copy(), excited.copy()))
        return joint_parameters, joint_covariance

    def population(
        xs: NDArray[np.float64],
        pdfs: NDArray[np.float64],
        *args: object,
        **kwargs: object,
    ):
        index = len(population_inputs)
        population_inputs.append((xs.copy(), pdfs.copy()))
        return populations[index], covariances[index]

    monkeypatch.setattr(ge_fitting, "fit_singleshot", joint)
    monkeypatch.setattr(ge_fitting, "fit_singleshot_p0", population)
    plots = Plots(NonPresentingHost())
    try:
        result = GE_Exp().analyze(
            ge_source(), GEAnalyzeOptions(angle=0.0, align_t1=False), plots=plots
        )
        # Quality cannot introduce another optimizer invocation.
        assert len(joint_inputs) == 1
        assert len(population_inputs) == 2
        xs, ground, excited = joint_inputs[0]
        expected_joint = compute_fit_quality(
            np.concatenate([ground, excited]),
            np.concatenate(
                [
                    calc_population_pdf(xs, *joint_parameters),
                    calc_population_pdf(xs, -1.0, 1.0, 0.2, 0.1, 0.8, 0.5, 0.01),
                ]
            ),
            dict(
                zip(
                    ("sg", "se", "s", "p0_g", "p0_e", "p_avg", "length_ratio"),
                    joint_parameters,
                    strict=True,
                )
            ),
            joint_covariance,
        )
        assert result.fit_quality is not None
        assert set(result.fit_quality) == {"joint", "ground", "excited"}
        assert result.fit_quality["joint"] == expected_joint
        for index, stage in enumerate(("ground", "excited")):
            x_stage, data = population_inputs[index]
            p0_g, p0_e, ratio = populations[index]
            expected = compute_fit_quality(
                data,
                calc_population_pdf(x_stage, -1.0, 1.0, 0.2, p0_g, p0_e, 0.5, ratio),
                {"p0_g": p0_g, "p0_e": p0_e, "length_ratio": ratio},
                covariances[index],
            )
            assert result.fit_quality[stage] == expected
        assert result.fit_quality["ground"].relative_parameter_errors[
            "p0_g"
        ] == pytest.approx(0.1)
        assert result.fit_quality["excited"].relative_parameter_errors[
            "p0_g"
        ] == pytest.approx(0.4)
    finally:
        plots.finish()
        plots.release()


def test_explicit_source_can_fit_without_cfg() -> None:
    source = RunRecord[GE_Cfg, GE_Result](cfg=None, result=ge_source().result)
    notebook = NotebookAdapter(host=NonPresentingHost())(GE_Exp())

    record = notebook.analyze(GEAnalyzeOptions(length_ratio=0.01), source=source)

    assert record.source is source
    assert record.source.cfg is None
    assert record.result.g_center == pytest.approx(-1 - 0.4j, abs=0.1)
    assert record.result.e_center == pytest.approx(1 + 0.4j, abs=0.1)
    assert list(record.figures) == ["fit"]


@pytest.mark.parametrize("backend", ["pca", "center"])
@pytest.mark.parametrize("initial_state", ["ground", "excited"])
def test_fit_and_post_use_the_same_prepared_states_and_separate_named_figures(
    backend: Literal["pca", "center"],
    initial_state: Literal["ground", "excited"],
    tmp_path: Path,
) -> None:
    original = ge_source()
    raw = original.result
    signals = raw.signals if initial_state == "ground" else raw.signals[::-1].copy()
    result = GE_Result(signals, raw.shot_indices, raw.prepared_states)
    source = RunRecord(original.cfg, result)
    before = result.signals.copy()
    fit_plots = Plots(NonPresentingHost())
    analysis = GE_Exp().analyze(
        source,
        GEAnalyzeOptions(
            backend=backend, initial_state=initial_state, length_ratio=0.01
        ),
        plots=fit_plots,
    )
    fit_plots.finish()
    assert analysis.initial_state == initial_state
    assert analysis.g_center == pytest.approx(-1 - 0.4j, abs=0.1)
    assert analysis.e_center == pytest.approx(1 + 0.4j, abs=0.1)
    assert analysis.init_pops[0, 0] > 0.8
    assert analysis.init_pops[1, 1] > 0.8
    assert list(fit_plots) == ["fit"]
    post_plots = Plots(NonPresentingHost())
    post = GE_Exp().post_analyze(
        source, analysis, GEPostAnalyzeOptions(), plots=post_plots
    )
    post_plots.finish()
    np.testing.assert_allclose(post.confusion.matrix, np.eye(3), atol=0.06)
    assert list(post_plots) == ["post"]
    assert len(post_plots["post"].axes) == 4
    np.testing.assert_array_equal(result.signals, before)
    fit_plots["fit"].savefig(tmp_path / "fit.png")
    post_plots["post"].savefig(tmp_path / "post.png")
    assert (tmp_path / "fit.png").stat().st_size > 0
    assert (tmp_path / "post.png").stat().st_size > 0


def test_invalid_state_and_calibration_fail_before_post_artifact() -> None:
    source = ge_source()
    plots = Plots(NonPresentingHost())
    with pytest.raises(ValueError, match="Unknown initial state"):
        GE_Exp().analyze(
            source,
            GEAnalyzeOptions(initial_state=cast(Any, "other")),
            plots=plots,
        )
    primary = GEAnalysis(
        initial_state="ground",
        fidelity=0.95,
        theta=0.0,
        threshold=0.0,
        ge_s=-0.2,
        g_center=-1 - 0.4j,
        e_center=1 + 0.4j,
        init_pops=np.array([[0.9, 0.1], [0.1, 0.9]]),
    )
    with pytest.raises(ValueError, match="Invalid GE calibration"):
        GE_Exp().post_analyze(source, primary, GEPostAnalyzeOptions(), plots=plots)
    assert len(plots) == 0


def test_canonical_round_trip_keeps_prepared_axes_and_cfg(
    tmp_path: Path,
) -> None:
    source = ge_source()
    path = tmp_path / "ge.hdf5"
    exp = GE_Exp()
    exp.save(source, path)
    loaded = exp.load(path)
    np.testing.assert_array_equal(loaded.result.signals, source.result.signals)
    np.testing.assert_array_equal(
        loaded.result.shot_indices, source.result.shot_indices
    )
    np.testing.assert_array_equal(loaded.result.prepared_states, [0, 1])
    assert isinstance(loaded.cfg, GE_Cfg)
    assert loaded.cfg.shots == 6000
