# sim/ — physical simulation for the mock soc (mocksim)

**Last updated:** 2026-06-26 (ArbWaveform asset length and decimated envelope)

High-level cheat-sheet for `program/v2/sim/`. Read before touching this package.
Implementation detail lives in the code and its docstrings; this file is concept,
architecture, and design boundaries only.

## Purpose

Upgrade the mock soc from a white-noise stub into a **physically-realistic**
data source, so the whole stack (experiment run -> analyze -> recover) can be
validated end to end offline. A `SimParams` injected into `make_mock_soc(sim=...)`
makes the mock soc return I/Q that, when fitted by the real experiment analyze
code, recovers the injected physics (f_qubit, pi gain, T1, T2, detuning) — and,
via the per-shot two-blob model, the singleshot |g>/|e> discrimination fidelity.

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
3. **read out** dispersively as two state-conditioned blobs — `s_g = S21(rf_g)`
   (qubit in |g>) and `s_e = S21(rf_e)` (qubit in |e>) — where `rf_g` / `rf_e`
   are the dressed resonator frequencies at the operating flux; the reps axis
   then draws a per-shot `Bernoulli(P_e)` to pick `s_e` (excited) else `s_g`.

**One unified path serves accumulated and singleshot.** The per-shot Bernoulli
is not gated on any "singleshot mode": its reps-mean is
`(1−P_e)·s_g + P_e·s_e == S21(rf_g) + P_e·[S21(rf_e) − S21(rf_g)]` (the
`mixed_signal` blend), so the accumulated (reps-averaged) readout is **unchanged**
(zero regression), while `get_raw` exposes the two Gaussian blobs a singleshot
experiment classifies (`GE_Exp` → PCA + histogram on the host). The accumulated
path now carries genuine shot noise `~ sqrt(P_e(1−P_e)/reps)`, so a slow
low-contrast fit (e.g. echo T2) needs enough reps to average it down — only reps
(not snr, which scales the Gaussian readout noise) suppresses the Bernoulli shot
noise.

The qubit frequency `f_qubit` and the dressed resonator frequencies come from the
existing fluxonium physics (`FluxoniumPredictor`, `calculate_dispersive_vs_flux_fast`,
`HangerModel`); the sim package re-implements none of it.

**Flux-constant work is cached by operating point.** With the operating flux
constant *within one acquire*, `f_qubit` and `rf_g` / `rf_e` are the same for every
sweep point, so the engine computes them once per run and feeds `rf_g` / `rf_e`
into each point's S21 blend — the fluxonium eigensolve behind `resonator_freqs`
(the dominant cost) never runs per point.  Identical physical parameters at the
same reduced flux also reuse a small process-local hot cache for the expensive
`predict_freq` / dressed-resonator prediction.  The cache key is the explicit
`SimParams` physics/alignment plus reduced flux; it never falls back to `MetaDict`
or moves the mock operating flux to a convenience point.  The cache is valid only
because the flux is fixed *for that acquire*; a per-point operating flux would have
to move that call back into the loop.

**Cooperative cancellation.** Acquire-level `stop_checkers` are passed into
`SimEngine`.  The sweep-point loop and the Lorentzian detune ensemble loop check
them between physics evaluations and raise `SimCancelledError` when cancellation
is requested, so `MockQickSoc.poll_data` fails fast instead of returning empty data
while QICK's polling loop is still waiting for shots.

**FLUX-AWARE-MOCK — operating flux from a live device.** By default the operating
flux is pinned at reduced flux = 1.0 (R-3).  `SimParams.flux_device` opts into
reading it live: when set, `engine._operating_signal` resolves the named device
from `GlobalDeviceManager` (a deliberate cross-layer reach from `program/v2/sim`
into `device/`; no import cycle since `device/` never imports the sim package — the
import is lazy inside the function), requires it to be a `FakeDevice`, and maps its
current `value` through `value_to_flux` to the reduced operating flux.  This mirrors
the real rig's software flux sweep: the runner does **software-per-acquire** (set
the device value, then run one acquire), so the flux is constant within an acquire
(the cache invariant above holds) and a fresh `SimEngine` is built every acquire
(base `_attach_sim_engine`), so the device read is effectively "read the live flux
just before each acquisition" with no stale cross-acquire value.  The binding lives
on the soc's *internal* SimParams copy (copy-on-input in `MockQickSoc.__init__`):
`set_flux_device` mutates that copy via `with_updates`, never the caller's instance
— critical because the GUI mock-connect passes the shared `DEFAULT_SIMPARAM`
singleton.  Resolution is fail-fast (missing device / non-FakeDevice raises) but the
*binding* is permitted before the device is registered.  Grep `FLUX-AWARE-MOCK` for
every coupling point.

## Module map

- `params.py` — `SimParams`: the physical parameter container (EJ/EC/EL, flux
  alignment, T1/T2/T2_star/thermal_pop, bare_rf/g/Ql/Qi, snr, pi_gain_len, seed),
  with the `0 < T2_star ≤ T2 ≤ 2·T1` validators and the derived
  `inhomogeneous_rate` (Γ). Also carries `poll_latency` (seconds/element, default
  1e-7): synthetic pacing for `MockQickSoc.poll_data`, not physics — set to 0.0 to
  skip the sleep entirely (e.g. in tests). Data + validation only, no physics logic.
- `bloch.py` — leaf TLS optical-Bloch propagator: segment generator, `expm`
  propagator, `evolve`, ground/excited helpers. Imports nothing from the project.
- `lowering.py` — module tree -> Bloch timeline + readout plan for one sweep
  point. Owns the single-rotating-frame detuning (plus the engine's per-node
  `detune_offset` frame shift), shaped-pulse discretisation, deterministic Branch
  selection, and the dmem (non-uniform T1) register indirection. Does NOT compute
  f_qubit, acc_buf, noise, S21, or the detune ensemble (the engine owns that).
- `waveforms.py` — shared peak-normalized envelope sampling for both lowering and
  decimated readout. `ArbWaveform` uses the asset's stored reference time axis and
  asset duration (`time[-1]`) as its playback length; config no longer supplies a
  separate arb waveform length.
- `readout.py` — dispersive readout: physical quantities -> complex IQ.
  `resonator_freqs` is the eigensolve (flux -> `rf_g` / `rf_e`); `s21` is the pure,
  eigh-free per-state hanger response (the engine calls it twice per point for the
  `s_g` / `s_e` blobs), and `mixed_signal` is the population-weighted blend (the
  accumulated reps-mean) — both *take* `rf_g` / `rf_e` (so the engine computes the
  dressed freqs once). Also `value_to_flux`. No sweeps / timelines / acc_buf /
  noise / the per-shot Bernoulli draw (the engine owns that).
- `engine.py` — `SimEngine`: glue. Pins the operating point at reduced flux
  `Phi/Phi0 = 1.0` (R-3; no longer derived from the cfg `dev` map), computes
  f_qubit AND `rf_g` / `rf_e` there ONCE (flux-constant, with a hot cache for
  identical operating points), drives lowering -> bloch -> readout, caches the
  deterministic `(s_g, s_e, p_e)` blob grids, and per round draws a per-shot
  `Bernoulli(p_e)` to select a blob and adds per-shot Gaussian noise into the QICK
  `(*loop_dims, nreads, 2)` int64 buffer (snr / reps / rounds / seed; fresh
  Bernoulli + noise per round so software-averaging works). The
  reps-mean is the accumulated `mixed_signal` blend (zero regression); `get_raw`
  sees the two blobs. Owns the Lorentzian quasi-static detune
  ensemble: it averages `P_e` over a deterministic Gauss-Legendre quadrature in `δ`
  (lowering applies each node as a frame shift via `detune_offset`), so T2\* emerges
  without identifying sequences. A *driveless* timeline (no qubit pulse, e.g. pure
  onetone) skips the quadrature: with every `omega == 0` the Bloch z-row decouples
  from `δ`, so `P_e` is `δ`-independent and the ensemble mean equals one eval
  exactly — a mathematical identity, not a per-experiment split (R-1 intact).

## Design boundaries and known limits

- **D1 — no `SimParams` => white-noise fallback.** `make_mock_soc()` without a
  sim is the unchanged stub; the sim path is fully opt-in.
- **Single rotating frame.** The whole timeline lives in one frame whose carrier
  is the qubit control pulses' frequency. Idle segments carry the frame detuning
  (not 0), which is what makes Ramsey fringes appear. Qubit pulses that disagree
  in frequency at a point have no single frame -> fast-fail.
- **Waveform envelopes.** Const, gauss/drag, cosine, flat_top, and `ArbWaveform`
  pulses all lower through the same scalar envelope path. `ArbWaveform` assets are
  represented as `abs(I+jQ)` because the current TLS Bloch/readout model has one
  amplitude envelope plus a static pulse phase; time-varying quadrature phase in
  the asset is not modeled separately.
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
- **D2 — decimated / lookback supported (model A).** `acquire_decimated` on a sim
  soc routes through the engine just like `acquire`: the engine lowers the single
  lookback point, evolves the Bloch vector to `P_e`, and renders the time-domain
  trace via `decimated_trace` — the readout envelope scaled by the steady mixed
  S21, **shifted by `sim.timeFly`** (the readout time of flight). So the trace is
  ~0 for program-time `< timeFly` and the readout pulse envelope appears at
  `[timeFly, timeFly + pulse_cfg.waveform.length)`, giving lookback a physical
  rising edge whose position `analyze` recovers as the trig_offset (`≈ timeFly`).
  The ADC/readout window length (`ro_cfg.ro_length`) only determines the sampled
  time axis and may outlive the generator envelope. Model A has no resonator
  ring-up transient (the steady S21 applied per sample); the accumulated companion
  `acc_buf` stays white noise (lookback never reads it). Decimated needs a
  `PulseReadout` (its `pulse_cfg` defines the envelope); a `DirectReadout` or more
  than one readout fast-fails.
- **Deterministic Branch supported.** A `Branch` selected by a registered
  sweep-loop counter (e.g. g/e prep) lowers its chosen sub-sequence; the frame
  detuning recurses into the selected branch. Measurement-conditional branches,
  nested branches, and a readout inside a branch fast-fail (control flow that
  needs shot-level feedback is out of scope).
- **Q3 — `DressedLabelingError` fallback.** Where the fast dispersive labeling is
  ambiguous, `resonator_freqs` degrades deterministically to "no dispersive
  shift" (`rf_g = rf_e = bare_rf`) and warns, rather than crashing — a real
  measurement never raises at that physics edge.
- **Q1 — noise model.** `snr` is per single repetition; fresh Gaussian noise is
  drawn each round so averaging over reps*rounds improves the effective SNR by
  `sqrt(reps*rounds)`, as on hardware.
- **Per-shot Bernoulli (singleshot) — one unified path.** The reps axis draws a
  per-shot `Bernoulli(P_e)` selecting the `s_g` / `s_e` blob, not a broadcast
  mean; this is *not* gated on a singleshot mode. Its reps-mean is the
  `mixed_signal` blend, so the accumulated readout is unchanged, while `get_raw`
  exposes two Gaussian blobs (`GE_Exp` classifies them host-side via PCA +
  histogram). Two consequences: (1) the accumulated path now carries genuine shot
  noise `~ sqrt(P_e(1−P_e)/reps)`, so slow low-contrast fits (echo T2) need enough
  reps to average it down — only reps, not snr, suppresses it; (2) a DEFAULT
  snr=300 fully separates the blobs (fidelity ~ 1), so a *meaningful* singleshot
  fidelity test must lower snr (~5–10) to overlap the blobs.

## Mock soccfg gotchas (when driving real experiments)

- The mock soccfg's const / flat_top pulse-length *register* grid is too coarse
  for a hard length sweep to compile (`len_rabi` const/flat_top raises a
  resolution error); drive length-Rabi with a gauss pulse (soft-sweep path).
- **Folding is a `f mod f_dds` analyzer-axis effect only, not a physics
  constraint.** `SimEngine` works in *true (absolute) frequencies* throughout —
  `f_qubit` (from `predict_freq`) and the drive / readout tones are never folded
  inside the Bloch dynamics, so the simulated TLS evolution is correct regardless
  of where the tones sit. Folding happens *downstream*, when the analyzer labels
  its absolute frequency axis (`sweep2array` -> QICK `freq2reg`/`reg2freq`): with
  the gen `interpolation==1` QICK applies no Nyquist check, so a tone is reported
  at `f mod f_dds` (it is *not* an fs/2 reflection). The mock gen f_dds is
  12288 MHz, so the whole fluxonium working set — f01 (~4 GHz), the dressed
  resonator (~7 GHz), and 6 GHz-class readouts with several-hundred-MHz sweeps —
  stays below f_dds and is reported *un-folded*. A tone above f_dds (e.g.
  12588 -> 300 MHz) would alias by `f mod f_dds`. Folding only affects *absolute*
  frequency-axis labels; *relative* quantities (detuning, decay times, gain
  scaling, fringe frequency) are folding-invariant regardless. So a direct
  absolute frequency inject->recover is clean as long as the tone stays below
  f_dds (the integration tests' f01 and readout both do).

## Tests

- `tests/program/v2/sim/test_bloch.py`, `test_bloch_limits.py` — Bloch core +
  analytic limits (Rabi, Ramsey, echo refocus at the decoupled-detuning layer).
- `test_params.py`, `test_readout.py`, `test_lowering.py` — per-layer unit tests.
- `test_engine.py` — engine assembly + acquire dispatch (feature *shape*: D1
  regression, peak/dip, oscillation, decay, fringes, round hook, decimated trace),
  the singleshot per-shot blobs (get_raw clusters on the |g>/|e> centres, pi/2 puts
  ~half on the excited blob, reps-mean == accumulated readout), the Lorentzian
  dephasing gates (quadrature reproduces the analytic FID, echo refocuses to T2,
  Ramsey decays faster, Γ=0 zero-regression), and a deterministic Branch smoke. The
  coherence-envelope helpers run at `reps=2000` to average the per-shot shot noise.
- `test_integration.py` — cross-experiment inject -> recover: real experiment
  `run` + `analyze` recover the injected f_qubit / pi gain / gain scaling / T1 /
  T2 + detuning, the dephasing proof (echo -> T2 and Γ-insensitive, Ramsey -> T2\*,
  Ramsey faster than echo), and the singleshot `GE_Exp` recover (low snr -> blob
  centres / preparation populations / a real discrimination fidelity that improves
  with snr). The echo recovery runs at `reps=2000` to average the shot noise.
