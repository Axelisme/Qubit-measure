# Explicit acquisition environments

**Last updated:** 2026-10-27

**讀者：** framework-dev。模擬 acquisition 的 request、round 與 raw source 契約。

`SimulationAcquireMixin` enters once through `MyProgramV2`. With a
`SimulationSoc`, it admits a synchronous call, builds one `AcquireRequest`,
and supplies numeric round data to the existing QICK acquisition pipeline.
`SimulationBinding` optionally accepts `SimulationCallbacks(wait_ready=callback)`. Admission calls
it before consuming the acquire sequence or constructing a request. Failure
releases admission and propagates; the environment callback owns any optional
startup-compilation fallback. External timers must wait at the environment's
ready interface before starting if startup time should be excluded.
`SimulationBinding(..., timing=SimulationTiming(time_scale=0.0))` optionally paces each round to
`time_scale × program.shot_time_s(rounds=1, ...)` seconds, passing each
structural template's point durations and the acquire's reps and relaxation.
Timeline durations include readout, pre/post delays and pending ADC activity;
structural templates cover every scan point once. The binding subtracts snapshot
and execution wall time, then sleeps only for a positive remainder. Failures do
not wait. With cancellation, waits check Stop at intervals of at most 10 ms; the
integrated acquisition discards the interrupted round, while decimated
acquisition keeps its completed trace and stops before the next round.
A finite nonnegative `time_scale` of 1.0 follows physical time, 0.1 is ten times
faster, and zero disables pacing. Without an injected clock, zero also skips
round clock reads. Generic bindings default
to zero; FTF, single- and six-fluxonium environments default to 1.0. The environment
builder owns this policy, not the GUI. For a 2 us timeline with zero relaxation,
100 reps and 10 rounds target 2 ms total. For 25 points averaging 20 us plus
500 us relaxation, 100 reps target 1.3 s per round. Actual wall time can exceed
the target when numeric work dominates.
`SimulationBinding(..., callbacks=SimulationCallbacks(on_end=callback))` optionally observes each admitted
acquire after all software rounds, including partial or failed acquisitions.
The callback receives captured `RoundRequest` records and admitted wall seconds,
including pacing. It runs before admission releases. Observer errors warn at
most once and cannot mask acquisition failures, even under warning-as-error
filters. An omitted on_end avoids this observer and its acquire-level clocks. FTF uses the
callback to persist its own local routing reports; program knows no file format.
Other SoC objects use the original path. Integrated simulation configures the
compiled program and readouts without serializing DAC/weight arrays for no-op
transport. Requested uploads still validate int16 envelope and weight dtypes;
QICK retains first-compilation ownership. Hardware and decimated paths keep
their original configuration. Classification, hooks, tracker, normalization,
averaging and Stop still belong to that pipeline.

`SimulationBinding(..., timing=SimulationTiming(clock=SimulationClock(...)))` optionally tracks physical
time independently of pacing speed. One environment shares one clock. It adds
`max(0, unscaled physical round seconds - elapsed wall seconds)` after each
completed round, including pacing in elapsed time. Failed or cancelled rounds
add nothing. Between rounds and acquisitions, wall time advances naturally.
The clock's `monotonic` callable can supply synthetic wall seconds. Generic
bindings do not create a clock; single-fluxonium environments own one.
`SimulationCallbacks(on_begin=...)` runs after readiness and before request
construction. Single-fluxonium uses it to freeze losses once per acquisition.
Preparation failures propagate and release admission without consuming a sequence.
Hardware acquisition does not call the binding or this clock.

## Build and own an environment

An environment file is trusted Python, not a sandbox or a configuration DSL.
For example, save this as `noise_environment.py`:

```python
from zcu_tools.experiment.simulation import white_noise


def build_environment():
    return white_noise(
        std_adc=1.0, seed=27, n_gens=2, n_readouts=1, routes=((0, 0),)
    )
```

The caller imports the builder and owns the returned environment. GUI startup
scripts can install its board and register its built drivers through public
facets. The GUI does not load or select an environment. `SimulationSoc.is_mock`
is True, so consumers can forward the board's identity without importing the
simulation implementation.

Notebook callers import the same builder and use its returned objects:

```python
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from noise_environment import build_environment


env = build_environment()
try:
    readout = PulseReadoutCfg(
        gen_ch=0, ro_ch=0, nqz=1, freq=100.0, gain=0.1,
        waveform=ConstWaveformCfg(length=0.1), ro_length=0.1,
    )
    prog = ModularProgramV2(
        env.soccfg, reps=4,
        modules=[readout.build("readout")],
    )
    averaged = prog.acquire(env.soc, rounds=2, progress=False)
    raw = prog.get_raw()
finally:
    env.devices.close_all_devices()
```

`SimulationEnvironment` returns `soc`, matching `soccfg`, `devices` and a public
`description`. The builder owns cleanup before successful return. The caller
owns cleanup after return. GUI registration transfers each accepted driver's
lifetime to DeviceService; rejected candidates remain caller-owned. The caller
retains ownership of the board and other environment resources. Register their
cleanup with the startup script's on-close hook when needed. After callbacks and
operation drain, the GUI tries every registered driver once. Failed closes stay
registered with an error; they do not prevent GUI exit or trigger automatic retry.

## Compact timelines and statistics

Each sweep point has one interpreted timeline per acquire, not one per shot.
The adapter interprets the Module tree in vector mode and emits templates,
partitioning the grid only where event structure differs. Reps and rounds repeat
the same controls. A timeline that depends on rep index fails fast with no
per-shot fallback.

For an unchanged environment snapshot, FTF evolves the initial density once and
reuses the physical result across rounds. A changed snapshot requires a new
evolution. Each readout state's conditional mean response is computed once per
distinct readout content; populations combine these means linearly.

Integrated statistics draw noise once per (round, point), not per shot.
Multinomial state counts represent projection noise. Gaussian receiver noise
uses the covariance of the sum over reps. Environments deliver an independent
RawShots source with retained noiseless inputs and a local round identity.
Requesting shots calls that source without rerunning evolution or the backend.
Raw ADC values remain int64, and round indices select distinct sampling streams.

Python loop counts do not grow with shots or sweep points. Template construction,
grouping and sampling use NumPy arrays or numba kernels for point-dependent work.
Python iterates over structural templates and unique content, not individual
points or reps.

## Numeric contract and extension points

`core.simulation` owns immutable numeric records. No Module, driver, GUI object
or callback crosses this boundary; `RoundRequest.cancel` is a read-only `is_set()`
query, not a callback. Time is us, frequency MHz, phase degrees and
gain normalized DAC amplitude. The frontend interprets exact supported Module
classes, not arbitrary tProcessor instructions. Unknown subclasses, raw program setup/body/cleanup or make_program overrides, raw loop
hooks and undefined or
incorrectly typed registers fail before the backend produces any data. Errors
identify the concrete Module type/name, loop coordinates and relevant register.

Supported controls are Pulse, PulseReadout, MuxReadout, Delay, SoftDelay, DelayAuto,
LoadValue, LoadWord, Repeat, Branch, ScanWith, ComputedPulse, TableLengthPulse,
TablePulseReadout, Join and the existing reset wrappers. Join interleaves branches
in the same static minimum-time order as hardware emission and rejects
cross-branch register dependencies. Reset wrappers emit actual pulses only;
they do not reset a quantum state. LoadValue compression does not affect semantic
values. LoadWord keeps uint32 patterns until a named readout frequency/gain
consumer decodes them; scalar delay/count consumers cannot reinterpret words.

The frontend interprets the Module tree once over a NumPy grid of non-reps
points. `AcquireRequest.templates` partitions that grid into structural classes.
Each event slot has point arrays for quantized timing, carrier, phase, signed gain,
envelope selection and activity. Repeat expands to its fixed count or register
range bound, with inactive slots beyond each point's count. Branch selections
partition templates. Generator and per-read ADC routing checks use these templates.
Any consumed parameter that depends on reps raises before backend execution.

`materialize_point(point)` expands one point into a detached `PointTimeline` for
diagnostics. `PulseEvent` retains signed gain, carrier MHz, phase degrees, source,
nested iteration coordinates and a common phase origin of zero. A nonconstant
`PulseEnvelope` owns immutable complex samples on the actual DAC sample clock.
Consumers interpolate linearly, hold the final sample through pulse end and use
zero outside the event. No per-pulse peak normalization or solver grid is imposed.
Const uses a unity envelope (`None`). A flat-top with at least 10 ns of plateau
becomes adjacent up-ramp, half-gain constant and down-ramp slots. Short flat-tops
keep whole sampled-envelope candidates. Long plateaus never expand onto a fine
grid. Ramp boundary samples preserve the scalar envelope's linear interpolation.

MuxReadout lowers the real compiled mixmux tones into simultaneous EventSlots
on one generator, with one static PFB ADC slot per declared tone. DDS, gain,
phase and demodulation rounding come from QICK's compiled parameters. Tone order
is ADC declaration order; channels are not sorted. The fixed mux compile profile
is opt-in through `make_mux_mock_soccfg`, with 1..8 explicit ADC routes. The
ordinary `make_mock_soccfg` defaults and clocks do not change. Numeric environments
must separately bind every requested route and supply any physical mux response.

`ReadoutWindow` keeps ADC samples/rate independent of generator duration. Per-point
windows retain trigger emission order, demodulation MHz/degrees and the generator
association for each read. The acquire-level windows specify fixed ADC shapes;
new consumers use template arrays for point-dependent timing/demodulation.
`materialize_point` expands one detached point for diagnostics, excluding reps.
Environments, not this frontend, explicitly convert us/MHz into the independent
numeric APIs' ns/GHz.

ComputedPulse reserves the maximum candidate blocking length. TableLengthPulse
uses the selected quantized duration and the existing runtime timestamp policy,
without padding to the longest entry. TablePulseReadout first plays its initial
compiled template; only its named loop's exec-after hook prepares the next words.
The last update restores table entry zero for the next outer repetition. Register
and template state belongs to one frontend preparation, never to another acquire.
The binding retains detached immutable DAC envelopes in a 32 MiB LRU pool across
acquisitions. Every preparation checks current DAC content, normalization, clocks
and flat-top duration; names and program identities are not cross-acquire keys.
The budget includes keys, Python samples and containers. Eviction releases only
pool references, so existing requests keep their immutable envelopes unchanged.
Normal hardware compilation and the existing block_mode timing policy are unchanged.

- `AcquireRequest` fixes C-order `loop_dims`, `avg_level`, non-reps `point_dims`
  and `point_names`, `reps`, point templates, declared ADC order, trigger-order
  windows and private sampling identity. Every active generator and every per-read
  ADC route must be bound.
- `EnvironmentSnapshot` contains numeric native-unit device values and retained
  model/calibration identity. The environment captures it immediately before
  each round, not in getters or preparation. A successful round hook may change
  the next snapshot. It cannot change this acquire's controls or layout.
- `SimulationBinding(capture_snapshot, execute, backend_version=..., seed=...)`
  coordinates admission and retains `last_request` and `rounds` privately in
  memory. `execute(RoundRequest)` returns `IntegratedRound(statistics, raw)` for
  integrated mode or `RoundSignals` for decimated mode. Mode/result mismatches raise.
  `statistics` is a pure core `RoundStatistics`; `raw` satisfies the program-owned
  `RawShots.materialize(reps, avg_level)` protocol.
- Integrated statistics contain per-point totals over reps, shared state counts,
  conditional per-shot sums and receiver covariance. The adapter validates and
  detaches these arrays. Each SimulationSoc retains at most 64 KiB of successful
  conditional-sum/covariance validation certificates, keyed by shape, dtype and
  exact bytes, not borrowed array identity. Counts, layout and noisy totals are
  checked each round, and every returned array is detached even on a certificate
  hit. Certificates live with the SoC and do not cross the core/backend boundary.
  Fixed request layouts and integrated normalization metadata are prepared once
  per acquire; this never freezes snapshots or certifies later backend values.
  Without threshold, classification or trackers, it divides
  totals by reps and readout length, removes QICK offsets and moves reads first,
  without allocating `acc_buf`. Hooks receive the running mean of completed rounds.
  Threshold, classification, trackers and `get_raw()` materialize deterministic
  int64 shots from the retained last-round source. The environment owns signal,
  noise, rounding and loop layout. Every source call returns fresh arrays with
  shape `(*loop_dims, reads, 2)`, keyed by requested ADC indices. Sources retain
  their noiseless inputs beyond backend borrows, cache eviction and subsequent
  rounds. Equal inputs, identity and numeric/RNG versions reproduce shots.
  Wrong reps/layout, invalid values and overflow raise ValueError. Numeric solver
  failures may raise RuntimeError. Raw is not shifted to fit the independently
  drawn direct totals. Getters cache the detached last-round arrays and never
  rerun a backend.
- Decimated IQ has shape `(execution, read, sample, 2)` and contains float64 ADC
  samples. Last axis is I/Q. Backend arrays are borrowed until the adapter copies
  them. Missing, duplicate or extra ADCs, wrong shapes and nonfinite values fail.
- The adapter orders channels, sums companion raw from the same decimated trace,
  and applies nearest-even int64 rounding. Overflow fails instead of wrapping or
  clipping. QICK divides integrated sums by readout length and averages only at
  its existing levels. Decimated acquisition averages software rounds, not reps.

`white_noise(std_adc=1.0, seed=None, ...)` explicitly selects independent,
zero-mean Gaussian I/Q per channel/sample. `std_adc` is finite, nonnegative and
measured per ADC sample. Integrated sums use standard deviation
`std_adc * sqrt(samples)`. Compact totals include receiver variance and reps/12
rounding variance per component. Zero produces exact zero, without rounding noise.
It models no pulse response. The root seed is supplied or generated once.
Integrated totals use PCG64 with
`SeedSequence([root_seed, acquire_sequence, round_index, 0])`; deferred raw uses
stream suffix 1. Decimated draws retain the three-component seed sequence.
NumPy, backend and derivation versions are retained. Calls, including failed or stopped calls, consume distinct
sequences. Getters do not sample. Recreating an environment with the same seed
and call sequence reproduces results within the recorded library versions.

`step_rounds=True` fails before request construction or sequence admission for
both simulation acquisition modes. Reentry on the same program or bound SoC fails
before overwriting the active call. `finally` releases call-owned buffers and
admission on normal return or exception. Integrated Stop reaches evolution's
chunk checkpoints through `RoundRequest.cancel`, discards the interrupted round,
and retains completed rounds (or no result when none completed); warmup and
hardware acquisition are unchanged. Decimated Stop reaches pacing only, without
adding backend evolution cancellation; its completed-trace policy is unchanged.
ADC work and pacing sleeps remain synchronous.

Only `SimulationProvenance.identifier` is copied to public Run metadata as optional
`simulation_id`. Backend/RNG versions and private requests remain on the typed
binding. Seeds, device-round inputs and model content are not put in normal GUI
or MCP metadata. Persistent private replay is a separate delivery.

General program classes depend on the public mixin entry, not frontend, binding,
signal-packing or SoC internals. `.importlinter` enforces that direct dependency
boundary while allowing the entry to delegate internally. `experiment.simulation`
composes program and device capabilities. Numeric backends depend on core records,
not on program or GUI owners.
