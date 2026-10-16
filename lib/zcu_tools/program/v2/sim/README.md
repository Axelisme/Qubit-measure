# sim/ physical simulation for the mock soc

**Last updated:** 2026-10-16

## Purpose

A `SimParams` injected into `make_mock_soc(sim=...)` makes the mock soc
return I/Q that, when fitted by the real experiment analyze code, recovers the
injected physics (f_qubit, pi gain, T1, T2, detuning) and, via the per-shot
two-blob model, the singleshot |g>/|e> discrimination fidelity.

## Injection architecture

`MyProgramV2.acquire` detects a mock soc carrying `SimParams` (`soc._sim_params`)
and routes through the `SimEngine`: the engine pre-computes the per-round raw
`acc_buf` budget and stashes it on the soc, then the real round loop runs
unchanged. `start_readout` / `poll_data` serve the budget, and
`_process_accumulated` / `_summarize_accumulated` / `round_hook` /
`cancel_flag` / `get_raw` are all reused. With no `SimParams` the branch is
skipped and the white-noise fallback path remains byte-for-byte stable.
This is why injecting params and running a real experiment class exercises the
genuine acquire pipeline, not a parallel mock path.

The qubit frequency `f_qubit` and the dressed resonator frequencies come from the
existing fluxonium physics (`FluxoniumPredictor`, `calculate_dispersive_vs_flux_fast`,
`HangerModel`); the sim package re-implements none of it.

One unified path serves accumulated and singleshot. The per-shot Bernoulli
is not gated on any "singleshot mode": its reps-mean is
`(1−P_e)·s_g + P_e·s_e == S21(rf_g) + P_e·[S21(rf_e) − S21(rf_g)]` (the
`mixed_signal` blend), so the accumulated (reps-averaged) readout follows the same
mean path, while `get_raw` exposes the two Gaussian blobs a singleshot
experiment classifies (`GE_Exp` → PCA + histogram on the host). The accumulated
path carries genuine shot noise `~ sqrt(P_e(1−P_e)/reps)`, so a slow
low-contrast fit (e.g. echo T2) needs enough reps to average it down. Only reps
(not snr, which scales the Gaussian readout noise) suppresses the Bernoulli shot
noise.

Acquire-level `cancel_flag` stays owned by the real
round loop's `finish_round()` path, matching hardware semantics: stop and SNR
early-stop are checked at completed-round boundaries, not inside one round's mock
physics compute. `SimEngine` still accepts an engine-local `cancel_flag` for
direct/internal callers, but `MyProgramV2.acquire` does not feed acquire-level
flags into it.

The environment owner resolves and validates a FakeDevice
before passing its value reader to `MockQickSoc.set_flux_source`. Each acquire
borrows that reader when constructing SimEngine. The engine reads once for that
acquire and retains the operating point across its rounds. A later acquire reads
the current value again. Reader failures propagate rather than falling back.

With no reader, the operating point is reduced flux 1.0. Passing `None` explicitly
unbinds a source. White-noise mocks reject binding because they have no SimEngine.
SimParams contains data only; neither MockSoc nor SimEngine looks up a registry.
Each soc keeps its own copied parameters and source binding.

Numba is an optional large-work kernel, not a mandatory dependency.

`SimParams.seed` is the mock simulator's root seed. A `MockQickSoc` owns the
per-soc child-seed sequence for SimEngine-backed `acquire()` and
`acquire_decimated()` calls, so consecutive acquires on one soc do not replay the
same RNG stream. Creating a new mock soc with the same root seed and running the
same acquire sequence reproduces the same child streams bit-for-bit. `seed=None`
keeps the route non-deterministic; it does not synthesize a deterministic child
sequence.

## Owners

- [params.py](params.py): `SimParams`, data + validation only; no flux,
  predictor, sweep, or timeline logic.
- [lowering.py](lowering.py): module tree -> Bloch timeline + readout plan for one sweep point.
- [bloch.py](bloch.py): leaf TLS optical-Bloch propagator. Imports nothing from the project.
- [readout.py](readout.py): dispersive readout, physical quantities -> complex IQ.
- [engine.py](engine.py): `SimEngine`, drives lowering -> bloch -> readout.
- [control_flow.py](control_flow.py): resolves control flow at one sweep point.
- [frame.py](frame.py): gives drives at several carriers their phase-coherent phases in the reference rotating frame.
- [dmem.py](dmem.py): collects `LoadValue` / `LoadWord` tables and reads a table's value at a sweep point.
- [waveforms.py](waveforms.py): shared peak-normalized envelope sampling for both lowering and decimated readout.
- [errors.py](errors.py): `UnsupportedModuleError`, the fast-fail for any module the lowering cannot represent faithfully.

## Design boundaries and known limits

- Idle segments evolve in the frame of the first
  qubit control pulse's frequency and carry its detuning (not 0), which is what
  makes Ramsey fringes appear. Each drive uses its own carrier. When pulses at
  one point play at different frequencies, `frame.py` gives each drive its
  phase-coherent phase, with time measured from the start of the shot. Hardware
  DDS phase follows absolute tProc time, so the model matches hardware only when
  every carrier offset times the shot period is a whole number of cycles.
- With a `soccfg`, lowering quantizes scalar and swept
  durations through QICK before evolving the qubit. Without a
  `soccfg`, direct uncompiled lowering keeps requested times.
- Decoherence has
  two parts: a homogeneous rate (`Tφ`, folded with T1 into the Bloch `T2` an
  echo recovers) carried by the Bloch propagator, and an inhomogeneous
  Lorentzian quasi-static detune (HWHM `Γ = 1/T2_star − 1/T2`) the engine averages
  over. `SimParams` is parameterised by `T1` / `T2` (homogeneous, echo) /
  `T2_star` (Ramsey), with `0 < T2_star ≤ T2 ≤ 2·T1`. The
  refocusing is sequence-agnostic: the engine never identifies the pulse
  sequence. A Ramsey free evolution accumulates the un-refocused ensemble phase
  (extra `exp(−Γt)` → T2\*), while an echo π flip refocuses every static detune (→
  the homogeneous T2), purely from the π pulse plus the ensemble average.
- A round initializes the qubit
  state once at the Boltzmann equilibrium population from `Temp` and the operating
  qubit frequency. Each rep evolves through the lowered
  pre-readout timeline, is read out without stochastic collapse in the density-only
  model, then passively evolves for `ProgramV2Cfg.relax_delay` before the next rep
  starts. Short `relax_delay` therefore makes repeated pulses a continuous-rep
  experiment; tests or experiments that intend independent shots must set a long
  enough delay or use an active reset.
- A `Branch` selected by a registered
  sweep-loop counter (e.g. g/e prep) lowers its chosen sub-sequence; the frame
  detuning recurses into the selected branch. Measurement-conditional branches,
  nested branches, and a readout inside a branch fast-fail (control flow that
  needs shot-level feedback is out of scope).
- A `Repeat` unrolls its body back to
  back, and a `ComputedPulse` plays the candidate its index register selects; the
  frame detuning includes pulses inside both. A readout or `Branch` inside a `Repeat`
  fast-fails.
- `LoadWord` carries raw hardware words and is
  supported only where a consumer defines semantics; currently `PulseReadout`
  decodes frequency words through the active `soccfg`. Raw gain words and arbitrary
  raw-word consumers still fast-fail.
- Decimated / lookback supported (model A). Model A has no resonator
  ring-up transient (the steady S21 applied per sample).
- `ArbWaveform` assets are
  represented as `abs(I+jQ)` because the current TLS Bloch/readout model has one
  amplitude envelope plus a static pulse phase; time-varying quadrature phase in
  the asset is not modeled separately.
- Where the fast dispersive labeling is
  ambiguous, `resonator_freqs` degrades deterministically to "no dispersive
  shift" (`rf_g = rf_e = bare_rf`) and warns, rather than crashing.
- `SimEngine` works in true (absolute) frequencies throughout.
  Folding only affects absolute frequency-axis labels; relative quantities
  (detuning, decay times, gain scaling, fringe frequency) are folding-invariant regardless.

Readout 的跨 owner 契約見 [ADR-0074](../../../../../docs/adr/0074-flat-readout-module.md)。
