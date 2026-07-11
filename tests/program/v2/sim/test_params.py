"""Tests for SimParams — the physical parameter container for SimEngine.

Covers:
- Construction with all required fields and defaults.
- Pydantic validation: wrong types and out-of-contract values raise.
- Coherence validators: T2 <= 2*T1 and T2_star <= T2, including boundary equality.
- Derived helpers: inhomogeneous_rate formula, the gamma=0 boundary, and
  Boltzmann equilibrium population from Temp and operating qubit frequency.
- Round-trip from a params.json-shaped dict to SimParams (alignment check).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError
from zcu_tools.experiment.v2_gui.adapters._support.ctx_helpers import proper_flux_range
from zcu_tools.gui.cfg import (
    EvalValue,
    SweepValue,
)
from zcu_tools.gui.session.types import ExpContext
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM, SimParams
from zcu_tools.program.v2.sim.readout import resonator_freqs, s21, value_to_flux

# ---------------------------------------------------------------------------
# Minimal valid kwargs reused across tests
# ---------------------------------------------------------------------------
_VALID: dict = {
    "EJ": 8.5,
    "EC": 1.0,
    "EL": 0.5,
    "flux_period": 0.002,
    "flux_half": 0.001,
    "T1": 50.0,
    "T2": 30.0,
    "T2_star": 30.0,  # T2_star == T2 => gamma=0 (pure homogeneous; no inhomogeneous)
    "bare_rf": 7.2,
    "g": 0.08,
    "Ql": 5000.0,
    "Qi": 50000.0,  # Qi > Ql required; gives dip depth = 1 - 5000/50000 = 0.9
    "snr": 10.0,
    "pi_gain_len": 0.4,
}


def _numeric_sweep_bounds(sweep: SweepValue) -> tuple[float, float]:
    if isinstance(sweep.start, EvalValue) or isinstance(sweep.stop, EvalValue):
        raise AssertionError("default guide sweep must have numeric fallback bounds")
    return float(sweep.start), float(sweep.stop)


class TestSimParamsConstruction:
    def test_required_fields(self) -> None:
        p = SimParams(**_VALID)
        assert p.EJ == 8.5
        assert p.EC == 1.0
        assert p.EL == 0.5
        assert p.flux_period == 0.002
        assert p.flux_half == 0.001
        assert p.T1 == 50.0
        assert p.T2 == 30.0
        assert p.T2_star == 30.0
        assert p.bare_rf == 7.2
        assert p.g == 0.08
        assert p.Ql == 5000.0
        assert p.Qi == 50000.0
        assert p.snr == 10.0
        assert p.readout_gain_noise_per_gain == 0.03
        assert p.pi_gain_len == 0.4

    def test_optional_defaults(self) -> None:
        p = SimParams(**_VALID)
        assert p.flux_bias == 0.0
        assert p.Temp == 0.060
        assert p.readout_photons_per_gain2 == 100.0
        assert p.readout_decay_rate_per_us == 2.0
        assert p.readout_decay_threshold_ratio == 0.1
        assert p.readout_decay_exponent == 1.0
        assert p.seed is None
        # FLUX-AWARE-MOCK: flux_device defaults to None (fixed reduced flux = 1.0).
        assert p.flux_device is None

    def test_flux_device_explicit(self) -> None:
        # FLUX-AWARE-MOCK: flux_device names a device whose live value drives flux.
        p = SimParams(**_VALID, flux_device="flux")
        assert p.flux_device == "flux"

    def test_optional_fields_explicit(self) -> None:
        p = SimParams(**_VALID, flux_bias=0.0005, Temp=0.035, seed=42)
        assert p.flux_bias == 0.0005
        assert p.Temp == 0.035
        assert p.seed == 42

    def test_extra_fields_forbidden(self) -> None:
        # ConfigBase has extra="forbid"; pass via dict to avoid static type error
        with pytest.raises(ValidationError):
            SimParams(**{**_VALID, "unexpected_field": 999})  # type: ignore[arg-type]


class TestSimParamsValidation:
    def test_wrong_type_for_EJ_raises(self) -> None:
        bad = {**_VALID, "EJ": "not_a_float"}
        with pytest.raises(ValidationError):
            SimParams(**bad)

    def test_wrong_type_for_seed_raises(self) -> None:
        bad = {**_VALID, "seed": 3.14}  # seed must be int or None
        with pytest.raises(ValidationError):
            SimParams(**bad)

    def test_missing_required_field_raises(self) -> None:
        incomplete = {k: v for k, v in _VALID.items() if k != "EJ"}
        with pytest.raises(ValidationError):
            SimParams(**incomplete)

    @pytest.mark.parametrize("temp", [-1.0, float("nan"), float("inf")])
    def test_invalid_temp_raises(self, temp: float) -> None:
        with pytest.raises(ValidationError, match="Temp must be finite and >= 0.0 K"):
            SimParams(**{**_VALID, "Temp": temp})

    def test_qi_equal_to_ql_raises(self) -> None:
        # Qi == Ql implies Qc = infinity and 1/Qc = 0, which is unphysical
        # (no coupling). The validator must reject this boundary case.
        bad = {**_VALID, "Qi": _VALID["Ql"]}
        with pytest.raises(ValidationError, match="Qi must be greater than Ql"):
            SimParams(**bad)

    def test_qi_less_than_ql_raises(self) -> None:
        # Qi < Ql implies 1/Qc < 0, i.e. Qc < 0 — unphysical.
        bad = {**_VALID, "Qi": _VALID["Ql"] - 1.0}
        with pytest.raises(ValidationError, match="Qi must be greater than Ql"):
            SimParams(**bad)

    # --- coherence validators ---

    def test_t2_greater_than_2t1_raises(self) -> None:
        # T2 > 2*T1 violates the T1-limit upper bound on homogeneous T2.
        bad = {**_VALID, "T1": 10.0, "T2": 25.0, "T2_star": 25.0}  # 25 > 2*10
        with pytest.raises(ValidationError, match="T2 must be <= 2\\*T1"):
            SimParams(**bad)

    def test_t2_equal_to_2t1_passes(self) -> None:
        # T2 == 2*T1 is the T1-limit (Tφ → ∞); equality is allowed.
        ok = {**_VALID, "T1": 15.0, "T2": 30.0, "T2_star": 30.0}  # 30 == 2*15
        p = SimParams(**ok)
        assert p.T2 == 30.0

    def test_t2_star_greater_than_t2_raises(self) -> None:
        # T2_star > T2 would imply a negative inhomogeneous rate γ, which is unphysical.
        bad = {**_VALID, "T2": 20.0, "T2_star": 25.0}  # 25 > 20
        with pytest.raises(ValidationError, match="T2_star must be <= T2"):
            SimParams(**bad)

    def test_t2_star_equal_to_t2_passes(self) -> None:
        # T2_star == T2 is the pure homogeneous limit (γ = 0); equality is allowed.
        ok = {**_VALID, "T2": 30.0, "T2_star": 30.0}
        p = SimParams(**ok)
        assert p.T2_star == p.T2

    @pytest.mark.parametrize("snr", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_snr_raises(self, snr: float) -> None:
        with pytest.raises(ValidationError, match="snr must be finite and > 0.0"):
            SimParams(**{**_VALID, "snr": snr})

    @pytest.mark.parametrize(
        "readout_gain_noise_per_gain",
        [-1.0, float("nan"), float("inf")],
    )
    def test_invalid_readout_gain_noise_per_gain_raises(
        self, readout_gain_noise_per_gain: float
    ) -> None:
        with pytest.raises(
            ValidationError,
            match="readout_gain_noise_per_gain must be finite and >= 0.0",
        ):
            SimParams(
                **_VALID,
                readout_gain_noise_per_gain=readout_gain_noise_per_gain,
            )

    @pytest.mark.parametrize(
        "readout_photons_per_gain2",
        [None, 0.0, -1.0, float("nan"), float("inf")],
    )
    def test_invalid_readout_photons_per_gain2_raises(
        self, readout_photons_per_gain2: float | None
    ) -> None:
        bad = {**_VALID, "readout_photons_per_gain2": readout_photons_per_gain2}
        with pytest.raises(ValidationError):
            SimParams(**bad)

    @pytest.mark.parametrize(
        "readout_decay_rate_per_us",
        [-1.0, float("nan"), float("inf")],
    )
    def test_invalid_readout_decay_rate_per_us_raises(
        self, readout_decay_rate_per_us: float
    ) -> None:
        with pytest.raises(
            ValidationError,
            match="readout_decay_rate_per_us must be finite and >= 0.0",
        ):
            SimParams(
                **_VALID,
                readout_decay_rate_per_us=readout_decay_rate_per_us,
            )

    @pytest.mark.parametrize(
        "readout_decay_threshold_ratio",
        [-1.0, float("nan"), float("inf")],
    )
    def test_invalid_readout_decay_threshold_ratio_raises(
        self, readout_decay_threshold_ratio: float
    ) -> None:
        with pytest.raises(
            ValidationError,
            match="readout_decay_threshold_ratio must be finite and >= 0.0",
        ):
            SimParams(
                **_VALID,
                readout_decay_threshold_ratio=readout_decay_threshold_ratio,
            )

    @pytest.mark.parametrize(
        "readout_decay_exponent",
        [0.0, -1.0, float("nan"), float("inf")],
    )
    def test_invalid_readout_decay_exponent_raises(
        self, readout_decay_exponent: float
    ) -> None:
        with pytest.raises(
            ValidationError,
            match="readout_decay_exponent must be finite and > 0.0",
        ):
            SimParams(
                **_VALID,
                readout_decay_exponent=readout_decay_exponent,
            )


class TestSimParamsCoherenceHelpers:
    """Tests for derived coherence properties."""

    def test_inhomogeneous_rate_nonzero(self) -> None:
        # γ = 1/T2_star - 1/T2; verify the formula with known inputs.
        p = SimParams(**{**_VALID, "T2": 20.0, "T2_star": 10.0})
        expected = 1.0 / 10.0 - 1.0 / 20.0  # 0.1 - 0.05 = 0.05
        assert abs(p.inhomogeneous_rate - expected) < 1e-12

    def test_inhomogeneous_rate_zero_when_t2star_equals_t2(self) -> None:
        # When T2_star == T2, γ = 0 exactly (pure homogeneous limit).
        p = SimParams(**{**_VALID, "T2": 30.0, "T2_star": 30.0})
        assert p.inhomogeneous_rate == 0.0

    def test_inhomogeneous_rate_small_when_t2star_close_to_t2(self) -> None:
        # γ should be small but positive when T2_star is slightly below T2.
        p = SimParams(**{**_VALID, "T2": 30.0, "T2_star": 29.0})
        assert p.inhomogeneous_rate > 0.0
        # Upper bound: 1/T2_star (T2 -> infinity limit)
        assert p.inhomogeneous_rate < 1.0 / 29.0


class TestSimParamsTemperatureHelpers:
    """Tests for Temp -> equilibrium excited-state population."""

    def test_zero_temp_returns_ground_equilibrium(self) -> None:
        p = SimParams(**{**_VALID, "Temp": 0.0})
        assert p.equilibrium_excited_population(5.0) == 0.0

    def test_positive_temp_uses_boltzmann_distribution(self) -> None:
        p = SimParams(**{**_VALID, "Temp": 0.050})
        f_qubit_ghz = 4.2
        h_over_k_b_k_per_ghz = 0.04799243073366221
        expected_weight = math.exp(-h_over_k_b_k_per_ghz * f_qubit_ghz / p.Temp)
        expected = expected_weight / (1.0 + expected_weight)
        assert p.equilibrium_excited_population(f_qubit_ghz) == pytest.approx(expected)

    @pytest.mark.parametrize("f_qubit_ghz", [-1.0, float("nan"), float("inf")])
    def test_invalid_frequency_raises(self, f_qubit_ghz: float) -> None:
        p = SimParams(**{**_VALID, "Temp": 0.050})
        with pytest.raises(
            ValueError,
            match="f_qubit_ghz must be finite and >= 0.0",
        ):
            p.equilibrium_excited_population(f_qubit_ghz)


class TestSimParamsPollLatency:
    """poll_latency is mock pacing — not physics; validated >= 0."""

    def test_default_is_1e7(self) -> None:
        p = SimParams(**_VALID)
        assert p.poll_latency == 1e-7

    def test_zero_disables_sleep(self) -> None:
        # 0.0 is the explicit sentinel to skip time.sleep entirely.
        p = SimParams(**_VALID, poll_latency=0.0)
        assert p.poll_latency == 0.0

    def test_custom_positive_value(self) -> None:
        p = SimParams(**_VALID, poll_latency=1e-5)
        assert p.poll_latency == 1e-5

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="poll_latency must be >= 0.0"):
            SimParams(**_VALID, poll_latency=-1e-8)


class TestSimParamsParamsJsonRoundTrip:
    """Verify that a params.json-shaped dict can be unpacked into SimParams.

    The dict structure mirrors what persistance.py produces:
        {
            "fluxdep_fit": {
                "params": {"EJ": ..., "EC": ..., "EL": ...},
                "flux_half": ...,
                "flux_period": ...,
            },
            "dispersive": {
                "bare_rf": ...,
                "g": ...,
            },
        }

    This test validates that field names align so external loaders can unpack
    params.json data into SimParams without translation glue.
    """

    _PARAMS_JSON: dict = {
        "fluxdep_fit": {
            "params": {"EJ": 8.5, "EC": 1.0, "EL": 0.5},
            "flux_half": 0.001,
            "flux_period": 0.002,
            "flux_int": 0.0,  # present in real files; not used by SimParams
            "plot_transitions": {},
        },
        "dispersive": {
            "bare_rf": 7.2,
            "g": 0.08,
        },
    }

    def test_round_trip_from_params_json_dict(self) -> None:
        pj = self._PARAMS_JSON
        fluxdep = pj["fluxdep_fit"]
        dispersive = pj["dispersive"]

        p = SimParams(
            # fluxonium Hamiltonian — aligned with fluxdep_fit.params keys
            EJ=fluxdep["params"]["EJ"],
            EC=fluxdep["params"]["EC"],
            EL=fluxdep["params"]["EL"],
            # flux alignment — aligned with fluxdep_fit top-level keys
            flux_half=fluxdep["flux_half"],
            flux_period=fluxdep["flux_period"],
            # readout — aligned with dispersive section keys
            bare_rf=dispersive["bare_rf"],
            g=dispersive["g"],
            # coherence and calibration (not in params.json; supplied separately)
            T1=50.0,
            T2=30.0,
            T2_star=30.0,  # T2_star == T2 => gamma=0 (pure homogeneous)
            Ql=5000.0,
            Qi=50000.0,
            snr=10.0,
            pi_gain_len=0.4,
        )

        assert p.EJ == fluxdep["params"]["EJ"]
        assert p.EC == fluxdep["params"]["EC"]
        assert p.EL == fluxdep["params"]["EL"]
        assert p.flux_half == fluxdep["flux_half"]
        assert p.flux_period == fluxdep["flux_period"]
        assert p.bare_rf == dispersive["bare_rf"]
        assert p.g == dispersive["g"]


# ---------------------------------------------------------------------------
# DEFAULT_SIMPARAM — realistic mock flux_period for visible flux_dep maps
# ---------------------------------------------------------------------------


class TestDefaultSimParamFluxPeriod:
    """DEFAULT_SIMPARAM.flux_period must sit within the onetone/twotone guide sweep.

    The production flux-dep guide sweep must map across at least one full period
    so the mock 2D flux_dep plot shows visible dispersion. A flux_period much
    larger than the sweep span would cover only a tiny fraction of a period and
    produce a flat colour map.
    """

    def test_default_readout_blobs_are_visible_over_base_noise(self) -> None:
        guide_ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
        guide = proper_flux_range(guide_ctx, expts=2)
        guide_start, guide_stop = _numeric_sweep_bounds(guide)
        guide_center = 0.5 * (guide_start + guide_stop)
        p = DEFAULT_SIMPARAM
        reduced_flux = value_to_flux(p, guide_center)
        rf_g, rf_e = resonator_freqs(p, reduced_flux)

        probe = np.array([rf_g], dtype=np.float64)
        ground_blob = s21(p, probe, rf_g)[0]
        excited_blob = s21(p, probe, rf_e)[0]
        separation_sigma = abs(excited_blob - ground_blob) * p.snr
        assert separation_sigma > 1.0, (
            "|g>/<e> readout blobs are not separated above the base noise scale; "
            f"separation={separation_sigma:.3f} sigma"
        )

    def test_guide_sweep_covers_at_least_one_period(self) -> None:
        # value_to_flux(v) = (v + flux_bias - flux_half) / flux_period + 0.5
        # The guide sweep should span at least one period for visible dispersion.
        guide_ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
        guide = proper_flux_range(guide_ctx, expts=2)
        guide_start, guide_stop = _numeric_sweep_bounds(guide)
        p = DEFAULT_SIMPARAM
        flux_lo = (guide_start + p.flux_bias - p.flux_half) / p.flux_period + 0.5
        flux_hi = (guide_stop + p.flux_bias - p.flux_half) / p.flux_period + 0.5
        span = flux_hi - flux_lo
        assert span >= 1.0, (
            f"guide sweep [{guide_start:g}, {guide_stop:g}] covers only "
            f"{span:.3f} periods "
            f"(need >= 1.0 for visible dispersion); flux_period={p.flux_period}"
        )
