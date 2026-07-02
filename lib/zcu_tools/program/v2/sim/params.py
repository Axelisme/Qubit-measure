"""SimParams — physical parameter container for the SimEngine mock simulator.

This module defines only the data container; no simulation logic lives here.
SimEngine (P1-5) consumes SimParams to drive TLS density-matrix simulation in
place of real hardware acquisition.

Alignment with params.json (see zcu_tools.meta_tool.QubitParams):
    fluxdep_fit.params["EJ"]  -> SimParams.EJ
    fluxdep_fit.params["EC"]  -> SimParams.EC
    fluxdep_fit.params["EL"]  -> SimParams.EL
    fluxdep_fit.flux_half     -> SimParams.flux_half
    fluxdep_fit.flux_period   -> SimParams.flux_period
    fluxdep_fit.flux_int      -> (not used by SimEngine directly; omitted)
    dispersive.bare_rf        -> SimParams.bare_rf
    dispersive.g              -> SimParams.g

Alignment with FluxoniumPredictor.__init__ (simulate/fluxonium/predict.py):
    params  -> (EJ, EC, EL)  passed as a tuple
    flux_half   -> SimParams.flux_half
    flux_period -> SimParams.flux_period
    flux_bias   -> SimParams.flux_bias  (optional, default 0.0)
"""

from __future__ import annotations

import math

from pydantic import field_validator, model_validator

from zcu_tools.cfg_model import ConfigBase


class SimParams(ConfigBase):
    """Physical parameters for the TLS Bloch-equation mock simulator.

    All frequency fields are in GHz.  All time fields (T1, T2, T2_star) are in µs.
    The choice of µs follows the experiment-layer convention used throughout
    experiment/v2/autofluxdep/{t1,t2echo,t2ramsey}.py.

    Fields
    ------
    Qubit physics (fluxonium Hamiltonian):
        EJ, EC, EL : float
            Josephson, charging, and inductive energies in GHz.

    Flux alignment:
        flux_period : float
            Device-value span corresponding to one full flux quantum.
        flux_half : float
            Device value at the half-flux point (Φ = 0.5 Φ₀).
        flux_bias : float, optional
            Additional bias added to device value when converting to reduced
            flux via FluxoniumPredictor.value_to_flux.  Defaults to 0.0.

    Coherence:
        T1 : float
            Longitudinal relaxation time in µs.
        T2 : float
            Homogeneous (echo) T2 in µs.  This is the T2 an echo experiment
            recovers.  It captures pure dephasing (homogeneous broadening) via
            ``1/T2 = 1/(2·T1) + 1/Tφ``.  Upper bound: ``T2 ≤ 2·T1`` (T1 limit).
        T2_star : float
            Ramsey T2* in µs.  This is the T2 a Ramsey experiment recovers.
            Inhomogeneous broadening (quasi-static Lorentzian detuning, refocusable
            by echo) adds an extra decay rate γ = ``1/T2_star − 1/T2``, so
            ``T2_star ≤ T2``.  When ``T2_star == T2`` the inhomogeneous rate γ = 0
            (pure homogeneous limit).
        thermal_pop : float, optional
            Thermal excited-state population at equilibrium, in [0, 1].
            Defaults to 0.0 (zero temperature).

    Readout resonator:
        bare_rf : float
            Bare resonator frequency in GHz.
        g : float
            Dispersive coupling strength in GHz.
        Ql : float
            Loaded quality factor (dimensionless).
        Qi : float
            Internal quality factor (dimensionless).  The coupling Q follows
            from the hanger relation ``1/Qc = 1/Ql - 1/Qi``; dip depth =
            ``1 - Ql/Qi``.  Must satisfy ``Qi > Ql`` (otherwise Qc ≤ 0,
            which is unphysical).

    Noise and calibration:
        snr : float
            Base per-sample Gaussian readout-noise scale.  The accumulated SNR
            still emerges from the readout gain, integration length, and the
            gain-proportional noise term below.
        readout_gain_noise_per_gain : float, optional
            Per-sample Gaussian readout noise added in quadrature with the base
            noise, proportional to the compressed PulseReadout drive amplitude.
            Defaults to 0.0.
        timeFly : float, optional
            Time of flight in µs: the readout-signal propagation delay.  The
            readout pulse program plays from program ``t = 0``, but the signal is
            only received by the ADC ``timeFly`` later — so a decimated/lookback
            trace is ~0 for the first ``timeFly`` and the readout envelope appears
            shifted by it.  This gives the simulated lookback trace a physical
            rising edge whose position a lookback ``analyze`` recovers as the
            trig_offset.  Defaults to 0.5.
        pi_gain_len : float
            Gain × length product required for a π rotation (ground truth for
            both amp_rabi and len_rabi experiments).  SimEngine derives Ω via
            ``Ω = (π / pi_gain_len) · gain``, so that
            ``θ = Ω · length = π · gain · length / pi_gain_len``.  The length
            unit must be consistent with the pulse length unit used in the
            timeline.
        readout_photons_per_gain2 : float, optional
            Calibration from PulseReadout gain² to mean intracavity photon
            number.  Defaults to 100.0; ``None`` is rejected so the readout
            power calibration is explicit.
        seed : int or None, optional
            RNG seed for reproducible noise.  None means non-deterministic.
            Defaults to None.

    Runtime flux binding (FLUX-AWARE-MOCK):
        flux_device : str or None, optional
            Name of a connected device in ``GlobalDeviceManager`` whose live value
            sets the operating flux.  When set, the engine reads that device's
            value at acquire time and maps it through ``value_to_flux`` (using this
            SimParams' flux_half / flux_period / flux_bias) to the reduced flux for
            f_qubit / dispersive prediction; only a ``FakeDevice`` is supported.
            When None (default), the operating flux is fixed at reduced flux = 1.0.
            Defaults to None.
    """

    # --- qubit Hamiltonian (GHz) ---
    EJ: float
    EC: float
    EL: float

    # --- flux alignment ---
    flux_period: float
    flux_half: float
    flux_bias: float = 0.0

    # --- coherence (µs) ---
    T1: float
    # T2: homogeneous (echo) T2; enforced T2 <= 2*T1 by _validate_coherence.
    T2: float
    # T2_star: Ramsey T2*; enforced T2_star <= T2 by _validate_coherence.
    T2_star: float
    thermal_pop: float = 0.0

    # --- readout resonator (GHz) ---
    bare_rf: float
    g: float
    Ql: float
    # Qi > Ql enforced by _validate_qi_gt_ql; Qc is derived, not stored.
    Qi: float

    # --- mock pacing (not physics) ---
    # poll_latency: seconds of time.sleep per data element in MockQickSoc.poll_data.
    # This is purely synthetic pacing — it has no physical meaning and does not
    # affect the simulated IQ values.  Set to 0.0 to skip the sleep entirely (e.g.
    # in tests where wall-time matters but measurement realism does not).
    poll_latency: float = 1e-7

    # --- runtime flux binding (FLUX-AWARE-MOCK) ---
    # flux_device: name of a connected device in GlobalDeviceManager whose live
    # value drives the operating flux of the simulation.  When None (the default),
    # the engine pins the operating point at reduced flux = 1.0 (R-3); when set,
    # the engine reads that device's value at acquire time and
    # maps it through this SimParams' affine (value_to_flux) to the reduced flux
    # used for f_qubit / dispersive prediction.  Only a FakeDevice is supported as
    # the source (see engine._operating_signal).  This is a *runtime* binding, not
    # physics: the field carries no validation and DEFAULT_SIMPARAM leaves it None.
    flux_device: str | None = None

    # --- noise and calibration ---
    snr: float
    # readout_gain_noise_per_gain: per-sample Gaussian noise coefficient for the
    # readout-drive-proportional noise source.  It is combined in quadrature with
    # the snr-derived base noise inside SimEngine.
    readout_gain_noise_per_gain: float = 0.0
    # pi_gain_len: the gain×length invariant shared by amp_rabi (sweeps gain)
    # and len_rabi (sweeps length).  SimEngine uses Ω = π/pi_gain_len · gain.
    pi_gain_len: float
    # readout_photons_per_gain2: gain->photon calibration for nonlinear readout.
    # None is intentionally not accepted; the default mock calibration is
    # 100 photons / gain^2.
    readout_photons_per_gain2: float = 100.0
    # timeFly: readout time-of-flight (µs). The decimated/lookback trace places
    # the readout envelope at program-time == timeFly (the trace is ~0 before it),
    # giving the simulated lookback a physical rising edge to recover as trig_offset.
    timeFly: float = 0.5
    seed: int | None = None

    @field_validator("poll_latency")
    @classmethod
    def _validate_poll_latency(cls, v: float) -> float:
        # Negative latency is meaningless; 0.0 is the explicit "no sleep" sentinel.
        if v < 0.0:
            raise ValueError(
                f"poll_latency must be >= 0.0 (got {v}); use 0.0 to disable pacing"
            )
        return v

    @field_validator("snr")
    @classmethod
    def _validate_snr(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"snr must be finite and > 0.0 (got {v!r})")
        return v

    @field_validator("readout_gain_noise_per_gain")
    @classmethod
    def _validate_readout_gain_noise_per_gain(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(
                f"readout_gain_noise_per_gain must be finite and >= 0.0 (got {v!r})"
            )
        return v

    @field_validator("readout_photons_per_gain2")
    @classmethod
    def _validate_readout_photons_per_gain2(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"readout_photons_per_gain2 must be finite and > 0.0 (got {v!r})"
            )
        return v

    @model_validator(mode="after")
    def _validate_qi_gt_ql(self) -> SimParams:
        # Qi > Ql is required so that 1/Qc = 1/Ql - 1/Qi > 0 (physical Qc).
        if self.Qi <= self.Ql:
            raise ValueError(
                f"Qi must be greater than Ql (got Qi={self.Qi}, Ql={self.Ql}); "
                f"Qi ≤ Ql implies Qc ≤ 0, which is unphysical."
            )
        return self

    @model_validator(mode="after")
    def _validate_coherence(self) -> SimParams:
        # T2 <= 2*T1: the T1-limit sets the ceiling on homogeneous T2.
        if self.T2 > 2.0 * self.T1:
            raise ValueError(
                f"T2 must be <= 2*T1 (got T2={self.T2} µs, T1={self.T1} µs, "
                f"2*T1={2.0 * self.T1} µs); the T1-limit caps homogeneous T2."
            )
        # T2_star <= T2: inhomogeneous broadening can only add decay, never remove it.
        if self.T2_star > self.T2:
            raise ValueError(
                f"T2_star must be <= T2 (got T2_star={self.T2_star} µs, "
                f"T2={self.T2} µs); inhomogeneous broadening only accelerates decay."
            )
        return self

    @property
    def inhomogeneous_rate(self) -> float:
        """Inhomogeneous (quasi-static) dephasing rate γ in 1/µs.

        Defined as γ = 1/T2_star − 1/T2.  This is the extra decay rate that
        echo refocuses but Ramsey cannot: a Lorentzian quasi-static detuning
        distribution with this half-width at half-maximum.  γ = 0 when
        T2_star == T2 (pure homogeneous limit).
        """
        return 1.0 / self.T2_star - 1.0 / self.T2


# ---------------------------------------------------------------------------
# Dev-only default operating point
# ---------------------------------------------------------------------------

# Dev operating point for the GUI mock-connect path.  Nyquist folding is a
# ``f mod f_dds`` analyzer-axis effect only, not a physics constraint (sim/README
# Nyquist note): the engine works in true absolute frequencies throughout, so the
# f_qubit the fluxonium prediction lands on at the fixed operating flux (R-3) is
# harmless to the Bloch dynamics regardless of where it sits relative to f_dds.
# T2_star < T2 exercises the inhomogeneous dephasing split so both Ramsey
# (→ T2_star) and echo (→ T2) paths produce meaningful data.
#
# THIS IS DEV-ONLY.  It is wired into the GUI mock-connect path (connection.py)
# so that "Use MockSoc" / gui_soc_connect(kind='mock') returns physically-
# realistic data.  Do NOT change the make_mock_soc() default signature — it stays
# sim=None (white noise) so all direct callers in tests remain unaffected.
DEFAULT_SIMPARAM: SimParams = SimParams(
    EJ=4.0,
    EC=1.0,
    EL=1.0,
    # flux_period=5e-3 sits within the onetone/twotone flux_dep guide default
    # sweep range (~±4e-3).  With flux_half=0.0 and flux_bias=0.0, sweeping v in
    # [-4e-3, 4e-3] maps reduced_flux across ~1.6 periods (sweet spot at v=0 =
    # reduced_flux 0.5), producing a clearly visible flux-dependent dispersion in
    # the mock 2D plot instead of a near-flat map.
    flux_period=5e-3,
    flux_half=0.0,
    flux_bias=0.0,
    T1=20.0,
    T2=15.0,  # homogeneous (echo) T2; 15 <= 2*T1=40 ✓
    T2_star=8.0,  # Ramsey T2*; 8 <= T2=15 ✓ — non-zero inhomogeneous rate
    bare_rf=6.0,
    g=0.08,
    Ql=1000.0,
    Qi=50000.0,
    snr=300.0,
    pi_gain_len=0.4,
    seed=12345,
)
