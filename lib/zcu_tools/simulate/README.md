# Numeric simulation tools

**Last updated:** 2026-10-23，效能量測邊界與重跑入口

**讀者：** framework-dev。查閱數值模擬的責任邊界、快取契約與效能證據入口。

`evolution` and `adc` are independent numeric tools. Neither imports QICK,
DeviceManager, resources, GUI, YAML or an environment. The existing fluxonium
prediction tools remain separate. ADR-0072 defines the package dependency rules.

## Measured performance and reproduction

Call-local reuse of directional-loss operators and matrix elements reduced the
fixed single-fluxonium flux scan's resident-warm acquisition wall time from
5.718435 s to 4.505389 s, about 21%. CPU time fell from 5.654049 s to
3.156055 s. The comparison used revisions `b19d249d` and `66729542`, one
old/new pair with cProfile, 21 DC points × 41 frequencies and 100 repetitions.
It retained production pacing, solver precision and physical settings. Complete
cold and warm signals matched exactly. This does not establish a speedup for
other environments or workloads.

The source-namespaced Numba cache saved 5.5–17.4 s on the first acquisition in
a new process when matching sources, dependencies and signatures already had
machine code on disk. At revision `eef5edf5`, four separate empty-cache/disk-hit
pairs covered single-fluxonium and FTF flux scans and amplitude Rabi. These
acquisition timers excluded imports and environment construction and did not use
cProfile. Each pair represents a possible first measurement after launching a
GUI process. This is not GUI window startup time, and the four savings must not
be added. Changed sources, versions or new signatures can still require compile.

These scripts live in the separately retained `sim-bottleneck-opt` task evidence
bundle, not in a fresh clone or an installed benchmark CLI. Its `scratch/`
contains the following reproduction entry points:

- `baseline/README.md` and `baseline/run.sh` define the original single/FTF
  workloads through public experiment `run` and program acquire.
- `warm-readout/implementation/report.md` describes copying the existing
  `compare/run.sh` and inputs to a new evidence directory for the old/new pair.
- `cold-cache/report.md` and `cold-cache/run.sh` define the two-fresh-process
  comparison. Flux uses 21 × 41 points, reps 100; Rabi uses 41 gains, reps 200.
- `six-baseline/README.md` and `six-baseline/launch.sh` define six-tone Rabi
  and T1 plus the separate single-tone range and flux workloads.

Use the original host with the saved evidence bundle, or preserve the bundle and
adapt its host-specific paths before rerunning elsewhere. An orchestrator must
provide a live worktree at each measurement's source revision with a locked
quality environment. Set the script's lane override; archived lanes no longer
exist. The evidence instructions preserve original configurations, cache state,
heavy-slot coordination, single-thread limits and production pacing. Python
entry points use `uv run --directory <live-lane> --no-sync --`. The bundle's
`scratch/final-report.md` records the numbers, source manifests and limitations.
No benchmark CLI is provided by this module.

## Numba machine-code cache

The first `zcu_tools.simulate` import selects one process-wide Numba cache
directory before simulator dispatchers are created. Without `NUMBA_CACHE_DIR`,
it uses `user_cache_root()/numba/<digest>/`; the shared root honors
`ZCU_TOOLS_CACHE_DIR`. An existing `NUMBA_CACHE_DIR` becomes a parent:
`<NUMBA_CACHE_DIR>/zcu-simulate-<digest>/`. Set overrides before the first
simulator import. Changing them later requires a new process.

The SHA-256 namespace includes every simulator `.py` file's relative name and
contents, plus the Python, Numba, NumPy and SciPy versions. Changing an inline
callee therefore selects a new namespace even when its cached caller is
unchanged. Unreadable sources fail import rather than reuse unprotected code.
Numba also checks its own ABI and CPU keys. This requires readable installed
Python sources and assumes those sources remain fixed for the process lifetime.

This sets the process-global `NUMBA_CACHE_DIR` and fixes
`NUMBA_CACHE_LOCATOR_CLASSES` to Numba's built-in `UserProvidedCacheLocator`,
replacing any caller-selected locators. It then reloads Numba configuration.
Unavailable cache storage fails when a cached dispatcher is created instead of
falling back to unnamespaced in-tree or user-wide code. Later-created Numba
dispatchers in other packages also use the selected directory and locator policy;
already-created dispatchers keep their existing cache locations.
The namespace does not track other packages' sources, runtime monkeypatches,
or changes to external native libraries. It does not make those packages'
cross-module caches safe. Old namespaces remain on disk until the cache owner
removes them; importing the simulator does not prune them.

Existing `cache=True` dispatchers can load machine code on a later launch with
matching sources, versions and signatures. This does not persist physical
results, density matrices or noisy IQ. Engine model/prepare/result caches remain
process-local and retain their existing lifetimes. ADC raw-path dispatchers with
`cache=False` still compile in each process. No solver, kernel or precision
policy changes.

## Evolution and explicit populations

Import records and functions from `zcu_tools.simulate.evolution`.
`SecularModel` supplies energy-basis Hamiltonians, drive quadratures, directional
rates and pure dephasing. `ModelSource` provides these models at quantized flux
levels. Energies use H/h in GHz, time is ns and phase is radians.

`Engine.evolve` takes a `ModelSource` with the initial DC density and a sequence
of `TimelineBatch` records. It returns densities at readout start in the DC
energy basis, not an internal-step trajectory. FTF uses the segment engine's fast algorithms by
default. The engine segments controls, selects kernels and shares deterministic
preparations across points and acquires. It never carries quantum states into a
new shot. Numeric tolerances do not certify basis convergence or the physical
validity of a model.

`ControlSequence`, `Envelope` and `NumericModel` remain inputs to independent
continuous-flux and model tools. They do not define an acquisition backend.

Call `populations(state, target_basis)` explicitly before readout. Basis vectors
are ordered columns in common ambient coordinates. The result preserves target
state labels and can be recorded independently of ADC generation. For engine\nbatches, call `batch_populations(densities, source_basis, target_basis)`; it\nreturns a real array in target state order without clipping solver residuals.

## ADC readout

Import from `zcu_tools.simulate.adc`. Construct `ReadoutModel` with joint-state
labels, absolute frequencies for 1 or 2 cavities, their resonator decay rates and a
directional transition matrix indexed `[target, source]`. Supply `PopulationInput`
with exactly the same state order. The ADC does not convert density matrices.

`ADC.response` solves populations and state-conditioned first field moments for
all initial joint states together. The resonators share reciprocal co-located
two-port radiative damping through `v*v.T`, where `v=sqrt(kappa_external/2)`.
The outgoing forward field is `b_in-sum(v*exp(-i*phi)*alpha)`, with each
resonator's `mismatch_phase_rad` defaulting to zero. This retains correlations between
state and field. Substituting a population-averaged frequency does not implement
this model. Each pulse has finite support, so the same equations cover ringup
and ringdown. `SampleGrid` supplies physical lookback times and demodulation.
Populations refer to the earliest readout pulse start, or zero with no pulses.
Earlier samples have zero physical field and do not advance the initial mixture.
Receiver offset and noise still apply to those samples.
`Receiver.time_of_flight_ns` delays the physical output, including its absolute
carrier phase, before ADC demodulation. The ADC conjugates the demodulated field
to microwave I+iQ before applying receiver gain, phase, offset and noise. See the
[ADC contracts](adc/README.md) for delay, mismatch and cache semantics.

For batched constant pulses, population and homogeneous moment exponentials are
shared across carrier frequencies. A Sylvester solve supplies the driven block
without assuming diagonalizable rate matrices. Near-singular driven blocks use
the full matrix exponential. Content-defined `PiecewiseLinearEnvelope` inputs use
bounded affine-input convolution propagation; opaque callbacks retain DOP853.
Flat integrated contents with identical timing/grid coordinates share small
carrier tiles, retaining per-carrier numerical guards.

`ADC.sample_windows` shares one initial-state draw across several per-execution
windows, while receiver noise remains independent across windows.
`ADC.sample` draws an initial joint state and a continuous-time
Markov path from the directional rates per shot. Each jump changes the cavity
generator without resetting its field. Vacuum fields are prepared once for all
states, including shaped envelopes; only homogeneous cavity memory is propagated
per jump. All incident carriers superpose on that path. Grids in
`sample_windows` share it through ringdown and gaps. Batched `sample_traces`
requires physical readout batches and shares each shot's path across ADCs.
Receiver noise remains independent per sample and window. The result includes float64 traces and their exact discrete
sums from one realization. Offset is added once per sample. Conditional responses
and integrated round statistics remain deterministic first moments.
A zero or singular positive-semidefinite 2x2 I/Q noise covariance is valid.
`SamplingIdentity`, imported from `zcu_tools.core.simulation`, selects a local
PCG64 stream without changing global RNG state. The ADC and deferred integrated
raw generation share this identity format.

Environment code owns unit conversion from QICK us/MHz/degrees to ns/GHz/radians
and composes evolution, explicit populations and ADC. ADC and white_noise own
complete integrated raw through structurally typed sources. `simulate.rounding`
owns shared numeric nearest-even int64 conversion and shot-axis layout. Program
owns the RawShots protocol, channel declaration order and acquisition normalization.
Core statistics remain pure data. Deferred raw does no work during mean draws;
its full CTMC paths use a separate stream and are not shifted to direct totals. [`evolution`](evolution/README.md) uses the segment engine's fast
algorithms by default and retains prepared data in memory. It does not read or
write an offline U-table library or persistent density records. Private replay
recomputes from archived inputs under matching numeric and RNG versions.

## FTF circuits and noise

[`ftf`](ftf/README.md) builds the fixed-coordinate F1-SQUID-F2 joint model with
full charge couplings, native-before-projection n², directional multilevel baths
and parent virtual readout transitions. `spectra` owns shared scalar/array loss spectra in angular GHz/ns/K;
`baths` uses them for joint directional rates, while fluxonium coherence combines
both directions for equilibrium T1. Default Q models and caller-supplied Q,
impedance and admittance options retain their existing definitions;
`dephasing` owns both the shared Notebook pure-rate formulas and CP multilevel
Markov dephasing. These tools do not read YAML, drivers or program Modules.
`experiment.ftf` composes them for acquisition.

## Flux source lines

`lines.FluxLine` maps one source to any positive number of reduced-flux loops.
`FilteredFlux` requires a matching finite DC vector and applies each source's
causal delay and scalar filter before loop mixing. `DistortionTerm` adds signed
exponential weights and positive decay times to the base lowpass step response.
A zero base time constant retains instantaneous feedthrough; an empty term tuple
retains the original first-order response. Passive tails remain within a shot
without a cutoff or cross-shot memory. The planner owns conversion of this
continuous flux into evolution inputs; the line tool does not freeze readout.
