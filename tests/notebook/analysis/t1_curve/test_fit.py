from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import zcu_tools.notebook.analysis.t1_curve.fit as fit_mod
import zcu_tools.notebook.analysis.t1_curve.t1_curve_fit as public_fit_mod
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.t1_curve import T1FitParams, fit_t1_noise_params

_PARAMS = (3.469, 0.952, 0.582)


def test_t1_curve_fit_module_exports_public_api() -> None:
    assert public_fit_mod.T1FitParams is T1FitParams
    assert public_fit_mod.fit_t1_noise_params is fit_t1_noise_params


def _fake_t1_model(
    params: tuple[float, float, float],  # noqa: ARG001
    fluxs: NDArray[np.float64],
    noise_channels: Sequence[tuple[str, dict[str, float]]],
    Temp: float,
    **kwargs: Any,  # noqa: ARG001
) -> NDArray[np.float64]:
    opts = {name: values for name, values in noise_channels}
    fluxs = np.asarray(fluxs, dtype=np.float64)
    rate = (1.2 + np.sin(5.0 * fluxs)) * Temp * 1e-4
    if "t1_capacitive" in opts:
        q_cap = opts["t1_capacitive"]["Q_cap"]
        rate += (1.0 + 0.3 * fluxs) / q_cap
    if "t1_quasiparticle_tunneling" in opts:
        x_qp = opts["t1_quasiparticle_tunneling"]["x_qp"]
        rate += (0.3 + fluxs**2) * x_qp
    if "t1_inductive" in opts:
        q_ind = opts["t1_inductive"]["Q_ind"]
        rate += (2.0 - fluxs) * 20.0 / q_ind
    return 1.0 / rate


def _sample_fluxs() -> NDArray[np.float64]:
    return np.linspace(0.08, 0.92, 18, dtype=np.float64)


def test_fit_t1_noise_params_recovers_all_free_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fit_mod, "calculate_eff_t1_vs_flux_fast", _fake_t1_model)
    true_params = T1FitParams(Q_cap=7e5, x_qp=1.8e-6, Q_ind=2.5e7, Temp=0.055)
    fluxs = _sample_fluxs()
    T1s = _fake_t1_model(_PARAMS, fluxs, _noise_channels(true_params), true_params.Temp)

    result = fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=T1FitParams(Q_cap=5e5, x_qp=2.5e-6, Q_ind=4e7, Temp=0.08),
        bounds={"Temp": (20e-3, 120e-3)},
        cutoff=12,
        qub_dim=4,
    )

    assert result.success
    assert result.fixed == ()
    assert result.free == ("Q_cap", "x_qp", "Q_ind", "Temp")
    assert result.params.Q_cap == pytest.approx(true_params.Q_cap, rel=2e-3)
    assert result.params.x_qp == pytest.approx(true_params.x_qp, rel=2e-3)
    assert result.params.Q_ind == pytest.approx(true_params.Q_ind, rel=2e-3)
    assert result.params.Temp == pytest.approx(true_params.Temp, rel=2e-3)
    np.testing.assert_allclose(result.model_T1s, T1s, rtol=2e-4)


def test_fit_t1_noise_params_keeps_fixed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fit_mod, "calculate_eff_t1_vs_flux_fast", _fake_t1_model)
    true_params = T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06)
    fluxs = _sample_fluxs()
    T1s = _fake_t1_model(_PARAMS, fluxs, _noise_channels(true_params), true_params.Temp)

    result = fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=T1FitParams(Q_cap=5e5, x_qp=2.0e-6, Q_ind=3e7, Temp=0.06),
        fixed=("Q_ind", "Temp"),
    )

    assert result.fixed == ("Q_ind", "Temp")
    assert result.free == ("Q_cap", "x_qp")
    assert result.params.Q_ind == true_params.Q_ind
    assert result.params.Temp == true_params.Temp
    assert result.stderr.Q_ind == 0.0
    assert result.stderr.Temp == 0.0


def test_fit_t1_noise_params_uses_provided_noise_params_as_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_channels: list[Sequence[tuple[str, dict[str, float]]]] = []

    def capture_model(
        params: tuple[float, float, float],
        fluxs: NDArray[np.float64],
        noise_channels: Sequence[tuple[str, dict[str, float]]],
        Temp: float,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        captured_channels.append(noise_channels)
        return _fake_t1_model(params, fluxs, noise_channels, Temp, **kwargs)

    monkeypatch.setattr(fit_mod, "calculate_eff_t1_vs_flux_fast", capture_model)
    true_params = T1FitParams(Temp=0.055, Q_cap=7e5, x_qp=1.8e-6)
    fluxs = _sample_fluxs()
    T1s = _fake_t1_model(_PARAMS, fluxs, _noise_channels(true_params), true_params.Temp)

    result = fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=T1FitParams(Temp=0.08, Q_cap=5e5, x_qp=2.5e-6),
        bounds={"Temp": (20e-3, 120e-3)},
        cutoff=12,
        qub_dim=4,
    )

    assert result.success
    assert result.fixed == ()
    assert result.free == ("Q_cap", "x_qp", "Temp")
    assert result.params.Q_ind is None
    assert result.stderr.Q_ind is None
    assert result.params.Q_cap == pytest.approx(true_params.Q_cap, rel=2e-3)
    assert result.params.x_qp == pytest.approx(true_params.x_qp, rel=2e-3)
    assert result.params.Temp == pytest.approx(true_params.Temp, rel=2e-3)
    assert captured_channels
    assert all(
        "t1_inductive" not in {name for name, _ in noise_channels}
        for noise_channels in captured_channels
    )


def test_fit_t1_noise_params_accepts_x_qp_public_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def capture_model(
        params: tuple[float, float, float],
        fluxs: NDArray[np.float64],
        noise_channels: Sequence[tuple[str, dict[str, float]]],
        Temp: float,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        opts = {name: values for name, values in noise_channels}
        assert "Q_qp" not in opts["t1_quasiparticle_tunneling"]
        captured["x_qp"] = opts["t1_quasiparticle_tunneling"]["x_qp"]
        return _fake_t1_model(params, fluxs, noise_channels, Temp, **kwargs)

    monkeypatch.setattr(fit_mod, "calculate_eff_t1_vs_flux_fast", capture_model)
    init = T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06)
    fluxs = _sample_fluxs()
    T1s = _fake_t1_model(_PARAMS, fluxs, _noise_channels(init), init.Temp)

    result = fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=init,
        fixed=("Q_cap", "x_qp", "Q_ind", "Temp"),
    )

    assert result.params.x_qp == init.x_qp
    assert captured["x_qp"] == init.x_qp


def test_fit_t1_noise_params_all_fixed_skips_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fit_mod, "calculate_eff_t1_vs_flux_fast", _fake_t1_model)
    init = T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06)
    fluxs = _sample_fluxs()
    T1s = _fake_t1_model(_PARAMS, fluxs, _noise_channels(init), init.Temp)

    result = fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=init,
        fixed=("Q_cap", "x_qp", "Q_ind", "Temp"),
    )

    assert result.optimizer_result is None
    assert result.success
    assert result.message == "all parameters fixed"
    assert result.cost == pytest.approx(0.0, abs=1e-24)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-12)


def test_fit_t1_noise_params_updates_progress_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fit_mod, "calculate_eff_t1_vs_flux_fast", _fake_t1_model)
    bars: list[_RecordingProgressBar] = []

    def make_recording_pbar(*args: Any, **kwargs: Any) -> _RecordingProgressBar:
        bar = _RecordingProgressBar(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(fit_mod, "make_pbar", make_recording_pbar)
    init = T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06)
    fluxs = _sample_fluxs()
    T1s = _fake_t1_model(_PARAMS, fluxs, _noise_channels(init), init.Temp)

    fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=init,
        max_nfev=3,
        progress=True,
    )

    assert len(bars) == 1
    assert bars[0].total == 3
    assert bars[0].n > 0
    assert bars[0].closed
    assert any(desc.startswith("T1 fit cost=") for desc in bars[0].descriptions)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bounds": {"bad": (1.0, 2.0)}}, "unknown bound"),
        ({"fixed": ("bad",)}, "unknown fixed"),
        ({"bounds": {"Q_cap": (9e5, 1e6)}}, "within bounds"),
    ],
)
def test_fit_t1_noise_params_validates_params_and_bounds(
    kwargs: dict[str, Any], match: str
) -> None:
    fluxs = _sample_fluxs()
    T1s = np.full_like(fluxs, 1e5)

    with pytest.raises(ValueError, match=match):
        fit_t1_noise_params(
            fluxs,
            T1s,
            _PARAMS,
            init=T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("init", "kwargs", "match"),
    [
        (T1FitParams(Temp=0.06), {}, "at least one T1 noise parameter"),
        (
            T1FitParams(Temp=0.06, Q_cap=8e5, x_qp=1.5e-6),
            {"fixed": ("Q_ind",)},
            "inactive",
        ),
        (
            T1FitParams(Temp=0.06, Q_cap=8e5, x_qp=1.5e-6),
            {"bounds": {"Q_ind": (1.0e5, 1.0e10)}},
            "inactive",
        ),
    ],
)
def test_fit_t1_noise_params_validates_whitelist_metadata(
    init: T1FitParams, kwargs: dict[str, Any], match: str
) -> None:
    fluxs = _sample_fluxs()
    T1s = np.full_like(fluxs, 1e5)

    with pytest.raises(ValueError, match=match):
        fit_t1_noise_params(
            fluxs,
            T1s,
            _PARAMS,
            init=init,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("fluxs", "T1s", "T1errs", "match"),
    [
        (np.array([0.1, 0.2]), np.array([1.0]), None, "same shape"),
        (np.array([0.1]), np.array([-1.0]), None, "finite and positive"),
        (np.array([0.1]), np.array([1.0]), np.array([0.0]), "positive finite"),
        (np.array([0.1]), np.array([1.0]), np.array([np.inf]), "positive finite"),
    ],
)
def test_fit_t1_noise_params_validates_data(
    fluxs: NDArray[np.float64],
    T1s: NDArray[np.float64],
    T1errs: NDArray[np.float64] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_t1_noise_params(
            fluxs,
            T1s,
            _PARAMS,
            init=T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06),
            T1errs=T1errs,
        )


def _noise_channels(
    params: T1FitParams,
) -> list[tuple[str, dict[str, float]]]:
    channels: list[tuple[str, dict[str, float]]] = []
    if params.Q_cap is not None:
        channels.append(("t1_capacitive", {"Q_cap": params.Q_cap}))
    if params.x_qp is not None:
        channels.append(("t1_quasiparticle_tunneling", {"x_qp": params.x_qp}))
    if params.Q_ind is not None:
        channels.append(("t1_inductive", {"Q_ind": params.Q_ind}))
    return channels


class _RecordingProgressBar:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.total = kwargs.get("total")
        self.n: int | float = 0
        self.closed = False
        self.descriptions: list[str] = []
        if "desc" in kwargs:
            self.descriptions.append(str(kwargs["desc"]))

    def update(self, value: int | float = 1) -> None:
        self.n += value

    def set_description(self, description: str) -> None:
        self.descriptions.append(description)

    def close(self) -> None:
        self.closed = True
