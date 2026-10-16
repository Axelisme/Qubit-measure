# Numeric simulation tools

**Last updated:** 2026-10-17，segment engine and recomputed replay

`evolution` and `adc` are independent numeric tools. Neither imports QICK,
DeviceManager, resources, GUI, YAML or an environment. The existing fluxonium
prediction tools remain separate. ADR-0072 defines the package dependency rules.

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
labels, absolute resonator frequencies, two sets of resonator decay rates and a
directional transition matrix indexed `[target, source]`. Supply `PopulationInput`
with exactly the same state order. The ADC does not convert density matrices.

`ADC.response` solves populations and state-conditioned first field moments for
all initial joint states together. Two resonators share reciprocal co-located
two-port radiative damping through `v*v.T`, where `v=sqrt(kappa_external/2)`.
The outgoing forward field is `b_in-v.T*alpha`. This retains correlations between
state and field. Substituting a population-averaged frequency does not implement
this model. Each pulse has finite support, so the same equations cover ringup
and ringdown. `SampleGrid` supplies physical lookback times and demodulation.
Populations refer to the earliest readout pulse start, or zero with no pulses.
Earlier samples have zero physical field and do not advance the initial mixture.
Receiver offset and noise still apply to those samples.

`ADC.sample_windows` shares one initial-state draw across several per-execution
windows, while receiver noise remains independent across windows.
`ADC.sample` computes these conditional means once and draws one initial joint
state per shot, then adds independent per-sample receiver noise. It does not draw
jump times or solve an ODE per shot. The result includes float64 traces and their
exact discrete sums from one realization. Offset is added once per sample.
A zero or singular positive-semidefinite 2x2 I/Q noise covariance is valid.
`SamplingIdentity`, imported from `zcu_tools.core.simulation`, selects a local
PCG64 stream without changing global RNG state. The ADC and deferred integrated
raw generation share this identity format.

Environment code owns unit conversion from QICK us/MHz/degrees to ns/GHz/radians
and composes evolution, explicit populations and ADC. Program adapters still own
channel order, QICK buffer shapes, nearest-even int64 conversion and acquisition
normalization. [`evolution`](evolution/README.md) uses the segment engine's fast
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
Markov dephasing. `lines.FilteredFlux` applies causal source delay/lowpass before
loop mixing and retains passive tails within a shot. These tools do not read
YAML, drivers or program Modules. `experiment.ftf` composes them for acquisition.
