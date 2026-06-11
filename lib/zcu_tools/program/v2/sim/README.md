# sim/ — physical simulation for the mock soc (mocksim)

**Last updated:** 2026-06-11 (mocksim Phase 2 dephasing)

High-level cheat-sheet for `program/v2/sim/`. Read before touching this package.
Implementation detail lives in the code and its docstrings; this file is concept,
architecture, and design boundaries only.

## Purpose

Upgrade the mock soc from a white-noise stub into a **physically-realistic**
data source, so the whole stack (experiment run -> analyze -> recover) can be
validated end to end offline. A `SimParams` injected into `make_mock_soc(sim=...)`
makes the mock soc return I/Q that, when fitted by the real experiment analyze
code, recovers the injected physics (f_qubit, pi gain, T1, T2, detuning).

## Injection architecture (layer A, hybrid)

`MyProgramV2.acquire` detects a mock soc carrying `SimParams` (`soc._sim_params`)
and routes through the `SimEngine`: the engine pre-computes the per-round raw
`acc_buf` budget and stashes it on the soc, then the **real** round loop runs
unchanged — `start_readout` / `poll_data` serve the budget, and
`_process_accumulated` / `_summarize_accumulated` / `round_hook` /
`stop_checkers` / `get_raw` are all reused. With no `SimParams` the branch is
skipped and behaviour is byte-for-byte the prior real path (design boundary D1).
This is why injecting params and running a real experiment class exercises the
genuine acquire pipeline, not a parallel mock path.

## TLS Bloch density-matrix model

For each sweep point the engine drives this chain:

1. **lower** the semantic module tree (Pulse / Reset / Delay / Readout) into a
   list of piecewise-constant `bloch.Segment` plus a readout plan,
2. **evolve** the Bloch vector through those segments (4x4 augmented affine
   propagator via `expm`) to an excited population `P_e`,
3. **read out** dispersively: the accumulated signal is the population-weighted
   mixture `S21(rf_g) + P_e * [S21(rf_e) - S21(rf_g)]`, where `rf_g` / `rf_e`
   are the dressed resonator frequencies at the operating flux.

The qubit frequency `f_qubit` and the dressed resonator frequencies come from the
existing fluxonium physics (`FluxoniumPredictor`, `calculate_dispersive_vs_flux_fast`,
`HangerModel`); the sim package re-implements none of it.

## Module map

- `params.py` — `SimParams`: the physical parameter container (EJ/EC/EL, flux
  alignment, T1/T2/T2_star/thermal_pop, bare_rf/g/Ql/Qi, snr, pi_gain_len, seed),
  with the `0 < T2_star ≤ T2 ≤ 2·T1` validators and the derived
  `inhomogeneous_rate` (Γ). Data + validation only, no physics logic.
- `bloch.py` — leaf TLS optical-Bloch propagator: segment generator, `expm`
  propagator, `evolve`, ground/excited helpers. Imports nothing from the project.
- `lowering.py` — module tree -> Bloch timeline + readout plan for one sweep
  point. Owns the single-rotating-frame detuning (plus the engine's per-node
  `detune_offset` frame shift), shaped-pulse discretisation, deterministic Branch
  selection, and the dmem (non-uniform T1) register indirection. Does NOT compute
  f_qubit, acc_buf, noise, S21, or the detune ensemble (the engine owns that).
- `readout.py` — dispersive readout: physical quantities -> complex IQ
  (`value_to_flux`, `resonator_freqs`, `s21`, `mixed_signal`). No sweeps /
  timelines / acc_buf / noise.
- `engine.py` — `SimEngine`: glue. Walks every sweep point, computes flux ->
  f_qubit, drives lowering -> bloch -> readout, lays the per-point IQ into the
  QICK `(*loop_dims, nreads, 2)` int64 buffer, and adds per-shot Gaussian noise
  (snr / reps / rounds / seed; fresh noise per round so software-averaging works).
  Owns the Lorentzian quasi-static detune ensemble: it averages `P_e` over a
  deterministic Gauss-Legendre quadrature in `δ` (lowering applies each node as a
  frame shift via `detune_offset`), so T2\* emerges without identifying sequences.

## Design boundaries and known limits

- **D1 — no `SimParams` => white-noise fallback.** `make_mock_soc()` without a
  sim is the unchanged stub; the sim path is fully opt-in.
- **Single rotating frame.** The whole timeline lives in one frame whose carrier
  is the qubit control pulses' frequency. Idle segments carry the frame detuning
  (not 0), which is what makes Ramsey fringes appear. Qubit pulses that disagree
  in frequency at a point have no single frame -> fast-fail.
- **Dephasing = homogeneous + Lorentzian quasi-static detune.** Decoherence has
  two parts: a *homogeneous* rate (`Tφ`, folded with T1 into the Bloch `T2` an
  echo recovers) carried by the Bloch propagator, and an *inhomogeneous*
  Lorentzian quasi-static detune (HWHM `Γ = 1/T2_star − 1/T2`) the engine averages
  over. `SimParams` is parameterised by `T1` / `T2` (homogeneous, echo) /
  `T2_star` (Ramsey), with `0 < T2_star ≤ T2 ≤ 2·T1`. The engine integrates the
  Lorentzian ensemble with a deterministic Gauss-Legendre quadrature (substitution
  `δ = Γ·tanθ` makes the Lorentzian weight uniform on `θ`); `Γ = 0` collapses to a
  single `δ = 0` node and reproduces the no-dephasing path bit-for-bit. The
  refocusing is **sequence-agnostic**: the engine never identifies the pulse
  sequence — a Ramsey free evolution accumulates the un-refocused ensemble phase
  (extra `exp(−Γt)` → T2\*), while an echo π flip refocuses every static detune (→
  the homogeneous T2), purely from the π pulse plus the ensemble average.
- **D2 — decimated / lookback are Phase-1 fast-fail.** `acquire_decimated` on a
  sim soc raises; only the accumulated path is modelled.
- **Deterministic Branch supported.** A `Branch` selected by a registered
  sweep-loop counter (e.g. g/e prep) lowers its chosen sub-sequence; the frame
  detuning recurses into the selected branch. Measurement-conditional branches,
  nested branches, and a readout inside a branch fast-fail (control flow that
  needs shot-level feedback is out of scope).
- **Q3 — `DressedLabelingError` fallback.** Where the fast dispersive labeling is
  ambiguous, `resonator_freqs` degrades deterministically to "no dispersive
  shift" (`rf_g = rf_e = bare_rf`) and warns, rather than crashing — a real
  measurement never raises at that physics edge.
- **Q1 — noise model.** `snr` is per single repetition; fresh noise is drawn each
  round so averaging over reps*rounds improves the effective SNR by
  `sqrt(reps*rounds)`, as on hardware.

## Mock soccfg gotchas (when driving real experiments)

- The mock soccfg's const / flat_top pulse-length *register* grid is too coarse
  for a hard length sweep to compile (`len_rabi` const/flat_top raises a
  resolution error); drive length-Rabi with a gauss pulse (soft-sweep path).
- f_qubit above Nyquist (fs/2 = 3072 MHz) aliases through `sweep2array`, so a
  frequency fit recovers the folded image. Pick a sub-Nyquist operating point for
  a direct frequency inject->recover.

## Tests

- `tests/program/v2/sim/test_bloch.py`, `test_bloch_limits.py` — Bloch core +
  analytic limits (Rabi, Ramsey, echo refocus at the decoupled-detuning layer).
- `test_params.py`, `test_readout.py`, `test_lowering.py` — per-layer unit tests.
- `test_engine.py` — engine assembly + acquire dispatch (feature *shape*: D1
  regression, peak/dip, oscillation, decay, fringes, round hook, decimated D2),
  the Lorentzian dephasing gates (quadrature reproduces the analytic FID, echo
  refocuses to T2, Ramsey decays faster, Γ=0 zero-regression), and a deterministic
  Branch smoke.
- `test_integration.py` — cross-experiment inject -> recover: real experiment
  `run` + `analyze` recover the injected f_qubit / pi gain / gain scaling / T1 /
  T2 + detuning, plus the dephasing proof (echo -> T2 and Γ-insensitive, Ramsey ->
  T2\*, Ramsey faster than echo).
