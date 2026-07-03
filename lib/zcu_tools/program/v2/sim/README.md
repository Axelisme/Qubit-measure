# sim/ — physical simulation for the mock soc (mocksim)

**Last updated:** 2026-07-03 — LoadWord fast-fail boundary

High-level cheat-sheet for `program/v2/sim/`. Read before touching this package.
Implementation detail lives in the code and its docstrings; this file is concept,
architecture, and design boundaries only.

## Purpose

`SimParams` turns the mock soc into a **physically-realistic** data source, so the
whole stack (experiment run -> analyze -> recover) can be validated end to end
offline. A `SimParams` injected into `make_mock_soc(sim=...)` makes the mock soc
return I/Q that, when fitted by the real experiment analyze code, recovers the
injected physics (f_qubit, pi gain, T1, T2, detuning) — and, via the per-shot
two-blob model, the singleshot |g>/|e> discrimination fidelity.

## Injection architecture (layer A, hybrid)

`MyProgramV2.acquire` detects a mock soc carrying `SimParams` (`soc._sim_params`)
and routes through the `SimEngine`: the engine pre-computes the per-round raw
`acc_buf` budget and stashes it on the soc, then the **real** round loop runs
unchanged — `start_readout` / `poll_data` serve the budget, and
`_process_accumulated` / `_summarize_accumulated` / `round_hook` /
`stop_checkers` / `get_raw` are all reused. With no `SimParams` the branch is
skipped and the white-noise fallback path remains byte-for-byte stable (design
boundary D1).
This is why injecting params and running a real experiment class exercises the
genuine acquire pipeline, not a parallel mock path.

## TLS Bloch density-matrix model

For each sweep point the engine drives this chain:

1. **lower** the semantic module tree (Pulse / Reset / Delay / Readout) into a
   list of piecewise-constant `bloch.Segment` plus a readout plan,
2. **evolve** the Bloch vector through those segments (4x4 augmented affine
   propagator, with closed-form free-evolution segments) to an excited population
   `P_e`; within a round the
   engine carries each rep's post-readout state through `relax_delay` into the
   next rep,
3. **read out** dispersively as two state-conditioned blobs — `s_g = S21(rf_g)`
   (qubit in |g>) and `s_e = S21(rf_e)` (qubit in |e>) — where `rf_g` / `rf_e`
   are the dressed resonator frequencies at the operating flux; the reps axis
   then draws a per-shot `Bernoulli(P_e)` to pick `s_e` (excited) else `s_g`.

**One unified path serves accumulated and singleshot.** The per-shot Bernoulli
is not gated on any "singleshot mode": its reps-mean is
`(1−P_e)·s_g + P_e·s_e == S21(rf_g) + P_e·[S21(rf_e) − S21(rf_g)]` (the
`mixed_signal` blend), so the accumulated (reps-averaged) readout follows the same
mean path, while `get_raw` exposes the two Gaussian blobs a singleshot
experiment classifies (`GE_Exp` → PCA + histogram on the host). The accumulated
path carries genuine shot noise `~ sqrt(P_e(1−P_e)/reps)`, so a slow
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

**Simulation caches stay internal to `SimEngine`.** Within one signal-grid build,
the engine separates per-point readout blobs/scales from qubit-state evolution:
sweep points that change only readout parameters reuse the same
pre-readout/inter-shot Bloch propagators and the same rep-resolved `P_e` chain,
while qubit-drive sweeps still compute distinct chains because their propagators
differ. These readout/evolution/population caches are private `SimEngine`
implementation details, not seams exposed to callers or sibling modules.

**Mocksim CPU loops yield the process GIL cooperatively.** Autofluxdep RUN work is
submitted to a dedicated `QThread`, but a Python/C-extension-heavy mock simulator
still shares the same interpreter and can delay the Qt main thread's Python slots.
Long signal-grid loops use a time-based `sleep(0)` yield so queued UI progress and
redraw events can run while the worker continues computing. This is only a
responsiveness boundary; it does not make the mocksim path Qt-dependent or change
the generated data.

**Numba is an optional large-work kernel, not a mandatory dependency.** For
multi-node detune ensembles with many distinct qubit state chains, `SimEngine`
uses a sim-local numba kernel when the optional `client` dependency is installed;
otherwise it falls back to the numpy/scalar path. No-dephasing and readout-only
cached sweeps skip numba to avoid JIT/cache-load overhead. Python-level
ThreadPool/joblib and numba `prange` parallel paths are not enabled because
measured scheduling/pickle overhead exceeds the current kernel savings.

**Cooperative stop boundary.** Acquire-level `stop_checkers` stay owned by the
real round loop's `finish_round()` path, matching hardware semantics: Stop and
SNR early-stop are checked after a round completes, not inside one round's mock
physics compute.  `SimEngine` still has an engine-local cancellation hook for
direct/internal callers, but `MyProgramV2.acquire` does not feed acquire-level
checkers into it.

**Profiling is opt-in and observational.** `ZCU_AUTOFLUXDEP_PROFILE=1` enables
MockSoc timing logs for the GUI stutter investigation without changing the
simulation schedule or data path. The probes split `poll_data`, `compute_round`,
signal-grid construction, operating-point lookup, and population-chain work so a
run log can distinguish Qt redraw delay from worker-side mock physics compute.
Each record includes the Python thread label; signal-grid logs also include the
cooperative GIL yield count.

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
  alignment, T1/T2/T2_star/thermal_pop, bare_rf/g/Ql/Qi, base `snr`,
  `readout_gain_noise_per_gain`, pi_gain_len, explicit
  `readout_photons_per_gain2`, seed),
  with the `0 < T2_star ≤ T2 ≤ 2·T1` validators and the derived
  `inhomogeneous_rate` (Γ). Also carries `poll_latency` (seconds/element, default
  1e-7): synthetic pacing for `MockQickSoc.poll_data`, not physics — set to 0.0 to
  skip the sleep entirely (e.g. in tests). Data + validation only, no physics logic.
- `bloch.py` — leaf TLS optical-Bloch propagator: segment generator, segment
  propagator, `evolve`, ground/excited helpers. Undriven free-evolution segments
  use the same closed-form affine solution as the full matrix exponential. Imports
  nothing from the project.
- `lowering.py` — module tree -> Bloch timeline + readout plan for one sweep
  point. Owns the single-rotating-frame detuning (plus the engine's per-node
  `detune_offset` frame shift), shaped-pulse discretisation, deterministic Branch
  selection, and the scalar `LoadValue` dmem indirection used by non-uniform T1.
  Raw `LoadWord` tables fast-fail because the simulator has no physics contract for
  arbitrary hardware register words. Does NOT compute f_qubit, acc_buf, noise, S21,
  or the detune ensemble (the engine owns that).
- `waveforms.py` — shared peak-normalized envelope sampling for both lowering and
  decimated readout. `ArbWaveform` uses the asset's stored reference time axis and
  asset duration (`time[-1]`) as its playback length; config no longer supplies a
  separate arb waveform length.
- `readout.py` — dispersive readout: physical quantities -> complex IQ.
  `resonator_freqs` is the eigensolve (flux -> `rf_g` / `rf_e`); `s21` is the pure,
  eigh-free per-state hanger response (the engine calls it twice per point for the
  `s_g` / `s_e` blobs), and `mixed_signal` is the population-weighted blend (the
  accumulated reps-mean) — both *take* `rf_g` / `rf_e` (so the engine computes the
  dressed freqs once). Also owns pure readout helpers: dispersive critical photon
  number, explicit unitless-gain photon-ratio calibration, safe-zone drive
  compression, state-visibility compression, envelope-weighted effective signal
  samples, envelope-RMS readout noise scaling, and integrated Gaussian noise
  sample scaling. Also `value_to_flux`. No sweeps /
  timelines / acc_buf / random noise / the per-shot Bernoulli draw (the engine owns
  that).
- `engine.py` — `SimEngine`: glue. Pins the operating point at reduced flux
  `Phi/Phi0 = 1.0` (R-3; no longer derived from the cfg `dev` map), computes
  f_qubit AND `rf_g` / `rf_e` there ONCE (flux-constant, with a hot cache for
  identical operating points), drives lowering -> bloch -> readout, privately
  separates point readout blobs from qubit-state evolution so readout-only sweeps
  reuse deterministic propagators, caches the deterministic `(s_g, s_e)` blob
  grids plus readout integration scales and a rep-resolved `p_e` grid, then per
  round draws a per-shot `Bernoulli(p_e)` to
  select a blob, multiplies the deterministic signal by the compressed readout gain
  and effective signal samples, and adds quadrature-combined base plus
  gain-proportional Gaussian integrated noise into the QICK
  `(*loop_dims, nreads, 2)` int64 buffer (noise parameters / reps / rounds / seed;
  fresh Bernoulli + noise per round so software-averaging works). The
  reps-mean is the accumulated `mixed_signal` blend; `get_raw`
  sees the two blobs. Owns the Lorentzian quasi-static detune
  ensemble: it averages `P_e` over a deterministic Gauss-Legendre quadrature in `δ`
  (lowering applies each node as a frame shift via `detune_offset`), so T2\* emerges
  without identifying sequences. A *driveless* timeline (no qubit pulse, e.g. pure
  onetone) skips the quadrature: with every `omega == 0` the Bloch z-row decouples
  from `δ`, so `P_e` is `δ`-independent and the ensemble mean equals one eval
  exactly — a mathematical identity, not a per-experiment split (R-1 intact).
  The deterministic state chain reinitializes at each round boundary; inside a
  round, `relax_delay` passively evolves every detune node from one rep to the next.
  Single-node chains use a scalar fast path; multi-node detune ensembles use a
  batched numpy fallback or, for large unique-chain work, the optional numba
  recurrence kernel.

## Design boundaries and known limits

- **D1 — no `SimParams` => white-noise fallback.** `make_mock_soc()` without a
  sim uses the white-noise stub; the sim path is fully opt-in.
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
  trace via `decimated_trace` — the readout gain times the readout envelope scaled
  by the steady mixed S21, **shifted by `sim.timeFly + pulse_cfg.pre_delay`**. With
  zero pulse pre-delay, the trace is ~0 for program-time `< timeFly` and the
  readout pulse envelope appears at `[timeFly, timeFly + pulse_cfg.waveform.length)`,
  giving lookback a physical rising edge whose position `analyze` recovers as the
  trig_offset (`≈ timeFly`).
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
- **Register dmem support is scalar-only.** The simulator can recover
  uncompressed `LoadValue` tables when a semantic consumer such as `DelayAuto`
  interprets the register as a scalar. `LoadWord` carries raw hardware words and
  therefore fast-fails until a specific consuming module defines simulator
  semantics for those words.
- **Q3 — `DressedLabelingError` fallback.** Where the fast dispersive labeling is
  ambiguous, `resonator_freqs` degrades deterministically to "no dispersive
  shift" (`rf_g = rf_e = bare_rf`) and warns, rather than crashing — a real
  measurement never raises at that physics edge.
- **Readout integration model.** Accumulated and singleshot raw `acc_buf` entries
  are integrated ADC sums. The deterministic raw center is linear in readout gain
  and in the effective signal samples while `nbar/ncrit <= 0.1`; above that
  dispersive guardrail the readout drive is softly compressed and the |g>/|e> blob
  separation is compressed around its midpoint. `readout_photons_per_gain2` is the
  explicit mock calibration from PulseReadout gain² to intracavity photons; its
  default is `100.0`, and `None` is rejected so the power calibration cannot
  silently depend on the current critical photon number. DirectReadout has no
  explicit generator gain and stays on the linear path.
  For accumulated `PulseReadout`, effective signal samples are the overlap between
  the compiled ADC sample axis and the generator envelope after applying
  `trig_offset - timeFly - pulse_pre_delay`; a readout window that starts before
  the signal arrival only integrates the later portion of the envelope.
  The base Gaussian raw-noise standard deviation scales as
  `sqrt(compiled_readout_samples)`. The optional gain-proportional Gaussian source
  uses the same readout envelope sampled on the ADC axis and scales with the
  compressed PulseReadout drive amplitude; independent sources are combined in
  quadrature.
  The real QICK/tracker normalization still divides by compiled `ro["length"]`,
  so a full-window const readout has roughly length-independent normalized mean
  and normalized Gaussian noise that decreases as `1/sqrt(length)`.
- **Q1 — noise model.** `snr` is only the base per-sample Gaussian readout scale,
  not a fixed final SNR. PulseReadout may add a second Gaussian source through
  `readout_gain_noise_per_gain`; this term is proportional to the compressed
  readout drive amplitude and is combined with the base source in quadrature.
  Fresh Gaussian noise is drawn each round so averaging over reps*rounds improves
  the effective SNR by `sqrt(reps*rounds)`, as on hardware. Full-window length
  sweeps are still expected to prefer their upper bounds in this first-order
  model: resonator ring-up and trigger-alignment penalties are not modeled.
- **`relax_delay` carries state within a round.** A round initializes the qubit
  state once at thermal equilibrium.  Each rep evolves through the lowered
  pre-readout timeline, is read out without stochastic collapse in the density-only
  model, then passively evolves for `ProgramV2Cfg.relax_delay` before the next rep
  starts. Short `relax_delay` therefore makes repeated pulses a continuous-rep
  experiment; tests or experiments that intend independent shots must set a long
  enough delay or use an active reset.
- **Per-shot Bernoulli (singleshot) — one unified path.** The reps axis draws a
  per-shot `Bernoulli(P_e)` selecting the `s_g` / `s_e` blob, not a broadcast
  mean; this is *not* gated on a singleshot mode. Its reps-mean is the
  `mixed_signal` blend, so the accumulated readout follows the same mean path, while `get_raw`
  exposes two Gaussian blobs (`GE_Exp` classifies them host-side via PCA +
  histogram). Two consequences: (1) the accumulated path carries genuine shot
  noise `~ sqrt(P_e(1−P_e)/reps)`, so slow low-contrast fits (echo T2) need enough
  reps to average it down — only reps, not the Gaussian noise parameters,
  suppresses it; (2) a DEFAULT base snr=300 fully separates the blobs (fidelity
  ~ 1), so a *meaningful* singleshot fidelity test must lower snr (~5–10) to
  overlap the blobs.

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
  Ramsey decays faster, Γ=0 no-inhomogeneous-broadening), and a deterministic
  Branch smoke. The coherence-envelope helpers run at `reps=2000` to average the
  per-shot shot noise.
- `test_integration.py` — cross-experiment inject -> recover: real experiment
  `run` + `analyze` recover the injected f_qubit / pi gain / gain scaling / T1 /
  T2 + detuning, the dephasing proof (echo -> T2 and Γ-insensitive, Ramsey -> T2\*,
  Ramsey faster than echo), and the singleshot `GE_Exp` recover (low snr -> blob
  centres / preparation populations / a real discrimination fidelity that improves
  with snr). The echo recovery runs at `reps=2000` to average the shot noise.
