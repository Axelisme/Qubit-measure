# Numeric simulation tools

**Last updated:** 2026-11-21, ADC frequency images

**讀者：** framework-dev、agent。查閱數值模擬的責任邊界、快取契約與實測效能。

`evolution` and `adc` are independent numeric tools. Neither imports QICK,
DeviceManager, resources, GUI, YAML or an environment. The existing fluxonium
prediction tools remain separate. ADR-0072 defines the package dependency rules.

Before adding or changing a kernel, a guard, a price or a measurement scene, or
optimizing speed, follow the [simulator maintenance rules](maintenance.md).

## Acquisition boundary

`zcu_tools.simulate.boundary` owns the immutable numeric acquisition records
shared by the frontend and environments. Import `TimeBase`, `AcquireRequest`,
`RoundRequest`, timeline slots, signals and statistics from this module.
`TimeBase` owns the rational common clock and integer-tick conversion.
`ChargeDriveRequirement` carries caller-supplied charge-line requirements;
physical environments validate membership. Records define units, array shapes
and borrow lifetimes without importing drivers, Module objects or GUI code.
See the [program simulation contract](../program/v2/simulation/README.md) for
frontend admission, round execution and ADC transport.

`simulate` does not import any other `zcu_tools` package.
The `.importlinter` contract `simulate-closed` forbids these dependencies,
including indirect imports, without exceptions. C19 continues to forbid
`core` from depending on `simulate`. Boundary records have no compatibility
re-export at their former location.

## Frozen scene storage

`evolution` owns the versioned index and single-record numeric cache codec.
`FROZEN_PHYSICAL_VERSION` identifies physical decoding semantics.
`SCENE_CACHE_VERSION` identifies index and cache-directory format.
Its public APIs take explicit paths and verify complete record identities.
They do not discover Git roots, build device models or solve references.
Tools choose the repository-shared cache; tests-side producers select independent
solvers. See [frozen numeric inputs and references](evolution/README.md#frozen-numeric-inputs-and-references).
This I/O boundary stays within the package's dependency-closure contract.

## Numeric parameter drift

`zcu_tools.simulate.drift.LogOU` evolves a positive numeric parameter with exact
log-OU transitions at caller-supplied seconds, using a caller-owned NumPy RNG.
It does not read settings, clocks or files. Equal times consume no RNG. The
first sample returns the reference value and sets the time origin.
`simulators.single_fluxonium.environment.drift.CoherenceDrift` composes loss streams and
owns acquisition settings and private truth I/O.

## Measured performance

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

The cold-cache pairs ran each workload in two fresh processes, first with an
empty cache and then with a disk hit. Flux scans used 21 × 41 points and 100
repetitions; amplitude Rabi used 41 gains and 200 repetitions. All runs called
the public experiment `run` and program acquire with production pacing and one
native thread. Those historical measurement scripts are not part of the
repository; the revisions and workloads above record those comparisons.

Simulator benchmarks run at the `Engine.evolve` boundary, not through program
acquire. Kernel measurements use one segment with a fixed `families=` selection.
Sequence measurements use multi-segment `TimelineBatch` inputs to measure total
evolution time, excluding program compilation and ADC sampling. The
[family Pareto table](../../../tools/README.md#engine-family-pareto-table) records
family errors and timings. The acquisition comparisons above remain historical
observations, not the current benchmark procedure.

## Numba machine-code cache

Importing `value2flux` or `flux2value` from `zcu_tools.simulate` requires only
NumPy. This basic import does not load Numba or SciPy, scan simulator sources,
access cache storage, or set Numba policy.

The first import of `zcu_tools.simulate.evolution` or `zcu_tools.simulate.adc`
selects one process-wide Numba cache directory before either tool creates
its dispatchers or first compiles JIT code. Both entries import the same private
bootstrap module. Python serializes its initialization and retains only a
successful import, so multiple tools share one namespace without nesting it.
A failed bootstrap propagates its error and is retried on the next tool import.
Without `NUMBA_CACHE_DIR`, it uses `<user-cache-root>/numba/<digest>/`.
The simulator resolves this root with `_numba_cache.user_cache_root()` without
importing `utils.cache`. The debug capture writer uses the same resolver for
`simulator-debug/v1`; see [capture persistence](evolution/README.md#capture-persistence).
`ZCU_TOOLS_CACHE_DIR` must be a non-empty absolute path; invalid values raise
`ValueError`. Without this override, the root is
`platformdirs.user_cache_dir("zcu_tools", "zcu_tools")`. If platform lookup fails,
it uses `<tempdir>/.zcu_tools`. These rules match `utils.cache.user_cache_root`,
so GUI and session cache roots keep their existing locations.
An existing `NUMBA_CACHE_DIR` becomes a parent:
`<NUMBA_CACHE_DIR>/zcu-simulate-<digest>/`. Set overrides before the first
Numba tool import. Changing them later requires a new process.

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
removes them; importing a Numba tool does not prune them.

Existing `cache=True` dispatchers can load machine code on a later launch with
matching sources, versions and signatures. This does not persist physical
results, density matrices or noisy IQ. ModelSource models and caller-retained preparations remain
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
Single constant-tone plateaus below 30 MHz offer `slow_envelope` and `dop853`.
Dissipative DC compares the channel's M⁶ construction cost, amortized over exact
prepare-key uses in the acquire, with direct integration. Nonzero
carriers hold phase at each midpoint and apply full-Hamiltonian matrix
exponentials between secular bath half-steps. The default is 20 segments per
carrier period. Dissipative DC uses the exact constant Lindblad channel instead
of splitting the bath. Its channel stores O(M⁴) complex values for M levels.
Bath-free DC uses one full-duration unitary.
Set `numerics.slow_envelope_segments_per_period` in single-fluxonium or the
six-fluxonium environment to refine this grid. Population error against dense
ODE is accepted within 1e-2; the 21-level Q1 calibration measured about 1e-4 or
less. This is not a bound for arbitrary gains, durations or coherences.
Carriers at or above 30 MHz keep the existing dispatch.

Acquisition cancellation takes effect between evolution chunks, discards the
interrupted round and retains completed rounds; startup warmup has no acquire
cancel query.

`ControlSequence`, `Envelope` and `NumericModel` remain inputs to independent
continuous-flux and model tools. They do not define an acquisition backend.

Call `populations(state, target_basis)` explicitly before readout. Basis vectors
are ordered columns in common ambient coordinates. The result preserves target
state labels and can be recorded independently of ADC generation. For engine\nbatches, call `batch_populations(densities, source_basis, target_basis)`; it\nreturns a real array in target state order without clipping solver residuals.

## Quantum resonator mode

Import `ResonatorModeInput`, `ResonatorModeResult` and `build_resonator_mode` from
`zcu_tools.simulate.resonator_mode`. The builder diagonalizes the full native
qubit Hamiltonian with one Fock-truncated resonator before retaining states.
Supply the qubit H/h in GHz, the already weighted charge coupling in GHz,
and named Hermitian qubit drive operators in the same fixed coordinates.
Drive operators act per real envelope unit; imaginary quadratures are zero.
Also supply the resonator frequency in GHz, its energy-loss rate kappa in 1/ns,
parent-eigenbasis directional qubit rates in 1/ns, and a `FluxLevel`.
The photon cutoff defaults to N=3, with occupations 0 through N-1.

`retained_labels` explicitly lists bare `(q, k)` pairs.
Here q indexes the ascending parent eigenstates and k is photon occupation.
Each retained label must identify one dressed state with overlap probability
strictly greater than 0.5. Missing, repeated or insufficient-overlap labels
raise `ValueError` naming the label and overlap.
Invalid matrix shapes, nonfinite values, non-Hermitian operators and invalid
rates also raise `ValueError`.

The result's `model` is a joint `SecularModel` in ascending dressed energy order.
Its fixed coordinates are that dressed basis, so `eigenvectors` is identity.
`label_to_index[(q, k)]` locates an initial state or an energy;
`index_to_label[i]` gives the inverse.
For example, subtract the energies at `(2, 0)` and `(1, 0)` to get their
dressed transition frequency.
Resonator loss keeps only energy-lowering matrix elements of a.
Qubit loss projects each parent jump into the joint basis.
Pure dephasing is zero.
The model identity uses public `simulate.evolution.digest` on all input fields,
including numeric arrays, port names, retained labels, cutoff and flux level.
Canonical JSON sorts mapping keys, so drive insertion order does not affect it.

Use this builder only when the caller explicitly needs photons in the evolution
space. `SingleFluxoniumModel` with `DispersiveResonator` instead supplies the
dispersive shifts for a semiclassical cavity; it adds no photon states to
evolution. The builder does not choose a model adapter, remove duplicate loss
terms or connect readout and acquisition environments.
`SingleFluxoniumModel.joint_resonator_model` supplies native-cutoff fluxonium
inputs and intrinsic qubit T1 without duplicate Purcell or thermal-photon terms;
see the [fluxonium adapter](fluxonium/README.md#modelpy--單-fluxonium-演化與讀出模型).

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
The grid carries either an explicit physical frequency image or a raw DDS
congruence. Only the raw variant folds the signed detuning around the model's
reference frequency.
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
`SamplingIdentity`, imported from `zcu_tools.simulate.boundary`, selects a local
PCG64 stream without changing global RNG state. The ADC and deferred integrated
raw generation share this identity format.

Environment code owns unit conversion from QICK us/MHz/degrees to ns/GHz/radians
and composes evolution, explicit populations and ADC. ADC and white_noise own
complete integrated raw through structurally typed sources. `simulate.rounding`
owns shared numeric nearest-even int64 conversion and shot-axis layout. Program
owns the RawShots protocol, channel declaration order and acquisition normalization.
Boundary statistics remain pure data. Deferred raw does no work during mean draws;
its full CTMC paths use a separate stream and are not shifted to direct totals. [`evolution`](evolution/README.md) uses the segment engine's fast
algorithms by default and shares prepared data within one evolve. It does not read or
write an offline U-table library or persistent density records. The optional
[Engine capture](evolution/README.md#capture-persistence) records inputs and actual routes.

## FTF circuits and noise

[`ftf`](../simulators/ftf/physics/README.md) builds the fixed-coordinate F1-SQUID-F2 joint model with
full charge couplings, native-before-projection n², directional multilevel baths
and parent virtual readout transitions. `spectra` owns shared scalar/array loss spectra in angular GHz/ns/K;
`baths` uses them for joint directional rates, while fluxonium coherence combines
both directions for equilibrium T1. Default Q models and caller-supplied Q,
impedance and admittance options retain their existing definitions;
`dephasing` owns both the shared Notebook pure-rate formulas and CP multilevel
Markov dephasing. These tools do not read YAML, drivers or program Modules.
`simulators.ftf.environment` composes them for acquisition.

## Flux source lines

`lines.FluxLine` maps one source to any positive number of reduced-flux loops.
`FilteredFlux` requires a matching finite DC vector and applies each source's
causal delay and scalar filter before loop mixing. `DistortionTerm` adds signed
exponential weights and positive decay times to the base lowpass step response.
A zero base time constant retains instantaneous feedthrough; an empty term tuple
retains the original first-order response. Passive tails remain within a shot
without a cutoff or cross-shot memory. The planner owns conversion of this
continuous flux into evolution inputs; the line tool does not freeze readout.
