# FTF environment composition

**Last updated:** 2026-10-16

`ftf(path_or_settings)` builds the same `SimulationEnvironment` for GUI startup
and Notebook callers. It creates fixed-unit V/A `FakeDevice` sources in one
`DeviceManager`, applies their initial values once and binds the selected channel
counts to the fixed mock QICK profile. The returned manager owns cleanup. A
builder failure closes its resources and preserves both startup and close errors.
The executable example is `notebook_md/simulation/environment.py` with adjacent
`ftf.yaml`. The Notebook walkthrough is `notebook_md/single_qubit.md`.

`FTFSource.snapshot` captures current device values immediately before each round.
It checks the configured source units and limits without restoring startup values.
`execute` uses only the detached `RoundRequest`. Identical physical executions
share deterministic preparations within a round; repetitions add independent
state/noise draws. Optional `library=EvolutionLibrary(absolute_path)` injects
numeric persistence. Each round fixes published table versions once. The numeric
owner handles formats, table matching, misses, direct solves and saving. Source
composition does not parse files or choose per-interval algorithms.
No density, source filter state or receiver noise is inherited by another shot.

`plan_point` is an author diagnostic interface. It converts each execution's
QICK us/MHz/degrees into ns/GHz/radians, preserving signed complex waveforms and
common carrier phase. The flux-line owner supplies delayed support,
envelope and phase events to the numeric model, so a short delayed pulse in an
idle interval cannot disappear merely because the allowed integration step is
large. Passive tails remain continuous, without imposing a global fine grid.
Local numeric phase advances to the quantized support
start using the numeric owner's nearest-2-ns geometry, so rounding a pulse start
does not shift its common oscillator reference. The plan contains independent
numeric inputs, not QICK buffers. `FTFSource` explicitly calls `Evolution.evolve`, `populations`, then
`ADC.sample_windows`. The program adapter owns packing, rounding and averaging.

The receiver PSD is a two-sided white spectral density in ADC sample units²*ns.
Independent discrete samples have covariance PSD/dt_ns, where dt_ns comes from
that execution's ADC output sample rate. Receiver gain does not scale this noise.
Zero/singular PSD matrices are valid. Several windows share one joint-initial
state draw per shot; receiver noise remains independent between windows.

FTF supports one common terminal readout pulse with one generator and one ADC.
It rejects active gate/bias instructions during or after readout. It does not
reject a passive lowpass tail. Quantum evolution follows the actual tail up to
the common readout start. At that instant the plan builds Hro, its basis,
directional rates and all retained-state pulls at the actual flux, including
DC, offset and residual tail. The readout model freezes these values throughout
the window and ignores further tail changes. This is an approximation, not a
claim that the line has settled or that ADC supports time-dependent flux.
No settling tolerance, runtime warning or physical admission threshold is used.

The model assumes weak dispersive readout, independent weak-coupling Markov
secular baths and exponential echo-characteristic dephasing. Native/local/M
truncations and numerical tolerances remain user choices. They do not certify
thermal-state coverage, dispersive accuracy or convergence. Conditional mean
fields include multilevel relaxation/thermal transitions, not random jump times.
Single-shot variance includes the joint-initial mixture and receiver noise only.

Plans, populations and state labels are author-only data. Normal GUI/MCP Run
metadata carries I/Q and the existing opaque provenance ID, not model truth.
There is no TLS fallback, single-qubit preset or runtime table build. Private
persistent replay remains separate from numeric records.

Authors opt in with `ftf(settings, private_replay=absolute_directory)`. Successful
rounds save checked content-addressed private inputs through
`resources.simulation_replay`. Acquire content is shared; each round retains its
snapshot and sampling identity. Detached complex waveform samples and full
settings survive source-file edits. `replay_round(round_path)` runs only that
round, without drivers, earlier acquisitions or current YAML. Exact numerical
and RNG dependency versions must match. Backend `ftf-direct-v3` identifies
actual delayed-flux event partitioning and fixed replay source selection, in
addition to the common-origin phase conversion. Archives from v2 and earlier
fail explicitly rather than silently executing different numeric arithmetic.
Historical F1/F2 calibration, the single accepted conditional Ramsey point and
cold/warm cost evidence remain evidence for their original backend only. They
have not been rerun with v3; no new precision or performance claim follows.
EvolutionLibrary owns opaque manifests
that pin/check the original publications and exact records before any solve.
Missing/damaged references fail, even if current inputs could be solved again.
Numeric files remain in the explicitly selected library and must be retained.
No pickle, arbitrary callable or generated noise is stored. Private archives
contain truth; they are never normal RunRecord/native/RPC fields. Failed rounds
are not marked as successful archives; in-memory binding inputs remain available.

## Offline library

`build_database(yaml_path, library_path, FTFBuildPlan(...), compiler_directory)`
reads the same YAML and uses `plan_point` to construct the fixed-DC family. No
drivers, GUI or QICK program is created. The optional prebuild plan supplies a
plateau duration, fixed complex microwave ratios and common-scale interval.
Actual runtime waveforms still come from timelines; unmatched inputs go direct.
`database.main(argv)` is the thin CLI. The example adjacent
`notebook_md/simulation/build_database.py` selects its YAML and data paths.

Only static closed constant-envelope/frequency-zero intervals use U tables.
Nonzero baths, dynamic flux and other controls retain full direct evolution and
persistent exact records. Changing only baths does not invalidate U identity,
but it changes the direct density key. Round device values always affect the
actual model, even with unchanged YAML. No seed/noise or density crosses shots.
`FTFSource.evaluations` retains each round/group's actual content receipt for
the latest acquire. It resets when the acquire sequence changes and contains no
quantum/noise state. Normal GUI/MCP metadata does not expose these author records.
Costs must distinguish table use, direct misses and direct-record reuse.
Closed-system timings do not establish general nonzero-bath performance.
