# Explicit acquisition environments

**Last updated:** 2026-11-02，原生 length 查表與逐點 native timing

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
`time_scale × program.request_round_time_s(request)` seconds. That owner sums
integer template durations and relaxation, multiplies by reps, and delegates the
single seconds conversion to TimeBase. Aggregate totals may exceed int64 and the
per-request ns bound without rejecting a long acquisition.
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
Integrated `acquire(..., charge_requirements=(ChargeDriveRequirement(name, gen_ch), ...))`
passes named physical charge requirements through the simulation request. The
record lives in `core.simulation`; names are diagnostics, not dispatch keys.
Single-fluxonium validates membership before evolution. White noise accepts
requirements without physical checks because it has no physical line model.
Six-fluxonium and FTF reject nonempty requirements before evolution until they
support this contract. Empty requirements retain each environment's behavior.
The entry consumes this keyword without forwarding it on hardware calls; hardware
wiring roles remain the operator's responsibility. Requirements are call-local.
Other SoC objects use the original path. Integrated simulation configures the
compiled program and readouts without serializing DAC/weight arrays for no-op
transport. Requested uploads still validate int16 envelope and weight dtypes;
QICK retains first-compilation ownership. Hardware and decimated paths keep
their original configuration. Shot processing, hooks, tracker, normalization,
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

## Compiled request builder

`simulation.compiled.build_compiled_request(prog, mode, binding, sequence, routes=soc.routes)`
reads `prog.compiled_stream` and returns detached numeric slots without calling a backend.
Its optional `charge_requirements` keyword preserves caller requirements in the
request for environment-owned physical validation, without inferring drive roles.
It predecodes the snapshot once and evaluates scan points as independent array
lanes. `LOOP_START` provides resolved register/wave start and step words.
Its `tables` map associates WMEM word addresses and field indexes with rotated
DMEM offsets. Its `register_tables` map associates canonical register addresses
with `offset` and ordered integer `values`. The builder applies each point's
wave fields and time-register values from these definitions.
The metadata shape is owned by the [IR snapshot contract](../ir/README.md#pre-optimization-compile-snapshot).
The builder checks the definitions against the compiled initial words and loop
shells; missing keys or inconsistent metadata raise before request construction.
Reps remain request metadata and are not executed. Marked shot-local loops iterate across lanes with `index < n` masks,
including zero and register-driven counts, without expanding the instruction stream.
A static live-in guard rejects registers or memory that a shot reads before
initialization and also writes across points/reps, even for equal-valued or
unobserved updates. Reading the outer reps counter in the shot is also an
unsupported live-in; sweep indices and shot-local counters remain supported.
Prologue-only constants and shot-local initialization remain supported.
Declared table registers are supported live-ins when their initial values and
closing-shell loads match `register_tables`; arbitrary cross-shot writes remain
unsupported.
The builder also rejects unresolved readout routes. An ADC without a wired
generator route receives an explicit missing-route diagnostic with read and time.
Content after the terminal self-jump is rejected, including unreachable NOPs;
body, cleanup and loop NOPs remain supported.
For a runtime-controlled ADC, all overlapping carriers on the selected route must
resolve the ADC word to the same frequency image. No carrier or conflicting images
raise before a request is returned. Static-config ADCs use the full frequency from
the shared QICK `calc_ro_freq` adapter and do not require an overlapping carrier
for this image resolution.
Wave writes dispatch by `(port, tmux_ch)` across the program's declared generators
and tProc-controlled readouts. For a port with tmux selectors, conf bits 8–15
select the registered user with `(conf >> 8) & 0xFF`. Active lanes must select the same user;
lane-dependent selectors or an unregistered port/selector raise `ValueError`.
A user without `tmux_ch` owns its port alone and ignores these bits for dispatch.
Duplicate keys and mixing users with and without selectors on one port raise
`ambiguous wave output port`. Mux tone masks still use conf bits 0–7.

`DPORT_WR` triggers only declared `dport` readouts whose port and mask bit match.
`TRIG set` triggers each declared `tport` readout on its port. `TRIG clr` records
no read. Both trigger paths use `machine.ref` plus explicit TIME or register
`s14`, and feed the same read registration and per-shot trigger-count checks.
Lane-dependent trigger ports raise `ValueError`.

Runtime readout WPORT commands are instantaneous config updates.
Each readout channel's FIFO uses the maximum of the command timestamp and its
previous effective time, without
wave duration or pipeline delays. Each read uses the last update effective at its
window start. A different frequency or phase inside the open window raises before
request construction; an equal update is accepted. Missing start-time config also
raises. Shot-local configuration writes must supply every read on that port and
drain before the next shot. A port configured only in the prologue can remain
constant. A read cannot use a prologue or previous-shot config once its port is
mutable in the shot, even when later updates have equal values. Pending FIFO
updates cannot carry across points/reps.
This timing rule is a simulation model, not a sample-exact hardware claim.

`SimulationAcquireMixin` calls this builder for `TimelineProgramV2`,
passing the bound SoC's routes. Other SoC objects keep the
original QICK acquisition path.
The core interpreter supports register ALU operations, Z/S flags, DMEM/WMEM and
native sweep/reps shells, marked inner loops and register-selected branches.
Body boundaries distinguish acquisition axes from body-local Repeat counters,
including loops emitted during Module lowering after QICK loop discovery.
Inner iterations produce separate slots and retain their nested coordinates.
Branch dispatch follows compiled register/flag effects and only enters its selected
marked case. Raw jumps inside case bodies remain unsupported.
Each ADC must execute its declared static trigger count in every point/shot;
matching scan totals cannot compensate for a per-shot mismatch. A mismatch raises
with ADC channel, static count and actual count before returning a request.
A declared ADC with no trigger is rejected before request construction; the builder
does not drop the ADC or invent a generator route.
Table lookups and runtime timeline arithmetic use the same register/memory effects,
including selected waveform ends, timeline maxima, floors and reference debits.
ARITH uses signed A/D27, B18 and C32 inputs and rejects pre-adder overflow.
Its bounded result fits signed46; s3 exposes low32 and the configured core source
exposes low/high32. DIV uses unsigned32 quotient/remainder and rejects zero division.
These are simulation models, not claims about unconfirmed DSP or divider boundaries.

Ordinary axis_signal_gen_v6 commands start at the maximum of their timestamp and
the previous command's end on the same generator. Same-timestamp flat-top segments
therefore play consecutively. Generators keep independent schedules. This model
omits dispatcher/CDC/pipeline delays and does not claim sample-exact DAC edges.
Mux commands use their own tone/mask contract instead.
Ordinary zero-output commands emit no event but still occupy their command duration.
DDS output uses a unity envelope. Input output selection, periodic mode and phase
reset raise before request construction. Holding the final sample is accepted only
when that envelope sample is exactly zero. Phase reset would affect later commands,
which requires DDS phase state outside the current numeric event contract.
Do not infer support for every control form or registered experiment from this builder.

Simulation acquisition has one frontend. `build_compiled_request` reads the
pre-optimization compiled stream, including supported instruction effects from
native calls and Module lowering. This does not imply support for every native
operation. There is no runtime frontend selector or fallback. Module lowering
remains part of program construction; acquisition does not interpret the Module tree.

## IR pass behavior

The hardware compiler optimizes IR after capturing the stream consumed by
simulation. These pass rules describe compiler output, not a second simulation
frontend. Pass conditions, compiler output and optimization guards are documented
in the [IR pass reference](../../../../../docs/reference/ir-pass-behavior.md).

## Build and own an environment

An environment file is trusted Python, not a sandbox or a configuration DSL.
For example, save this as `noise_environment.py`:

```python
from pathlib import Path

import yaml
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program import build_soccfg


def build_environment():
    profile = yaml.safe_load(Path(__file__).with_name("profile.yaml").read_text())
    return white_noise(build_soccfg(profile), std_adc=1.0, seed=27, routes="all")
```

The caller imports the builder and owns the returned environment. GUI startup
scripts can install its board and register its built drivers through public
facets. The GUI does not load or select an environment. `SimulationSoc.is_mock`
is True, so consumers can forward the board's identity without importing the
simulation implementation.

Notebook callers import the same builder and use its returned objects:

```python
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from noise_environment import build_environment


env = build_environment()
try:
    readout = PulseReadoutCfg(
        gen_ch=0, ro_ch=0, nqz=1, freq=100.0, gain=0.1,
        waveform=ConstWaveformCfg(length=0.1), ro_length=0.1,
    )
    builder = TimelineBuilder(env.soccfg, reps=4)
    builder.emit(readout.build("readout"))
    prog = builder.build()
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
The adapter interprets the pre-optimization compiled stream over independent
scan lanes and emits numeric templates. Reps and rounds repeat
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
query, not a callback. Every request time is an integer tick in its required
`TimeBase`. Array times use int64 without changing point/read shapes. Frequency
is MHz, phase is degrees and gain is normalized DAC amplitude. The builder gathers
tProc, declared generator fabric/envelope and declared ADC output clocks from
soccfg. Undeclared channels do not participate. TimeBase belongs to the request;
adding a declared heterogeneous clock may change tick units, but must preserve
the original channel's physical time grids and output.
Start, duration, trigger, phase origin and sample periods remain integers through
serialization, FIFO admission, window overlap and body-end accounting. Relax delay
uses QICK's compiled tProc-cycle rounding before mapping to the common clock. The frontend accepts supported instruction effects
from native operations, Module lowering and raw assembly, including NOP hooks.
It rejects measurement-dependent jumps, unmarked jumps, body waits/resyncs,
tProc clock reads and cross-shot state carry before backend execution.
Errors identify the instruction class, pre-optimization position and available
loop, case and source context.

Supported controls are Pulse, PulseReadout, MuxReadout, Delay, SoftDelay, DelayAuto,
LoadValue, LoadWord, native repeat/branch, ComputedPulse,
and the existing reset wrappers. Parallel sequences use native
fork/join and preserve compiled emission order. Their register effects follow the
same shot-local and live-in rules as other instructions. Reset wrappers emit
actual pulses only; they do not reset a quantum state. LoadValue compression
does not affect semantic values.
LoadWord preserves uint32 words; each compiled consumer applies its operand semantics.

The frontend predecodes the compiled stream once over a NumPy grid of non-reps
points. `AcquireRequest.templates` covers that grid with numeric slots.
Each event slot has point arrays for quantized timing, carrier, phase, signed gain,
envelope selection and activity. Marked inner loops execute with active-lane masks
and retain nested iteration coordinates. Branches select lanes using compiled
registers and flags. Generator and per-read ADC routing checks use these templates.
Reps remain request metadata; their shell cannot update other shot state.

`materialize_point(point)` expands one point into a detached `PointTimeline` for
diagnostics. `PulseEvent` retains signed gain, carrier MHz, phase degrees, source,
nested iteration coordinates and a common phase origin of zero. A nonconstant
`PulseEnvelope` owns immutable complex samples on the actual DAC sample clock.
Consumers interpolate linearly, hold the final sample through pulse end and use
zero outside the event. No per-pulse peak normalization or solver grid is imposed.
Const uses a unity envelope (`None`). Compiled waveform commands determine event
durations and envelope samples. Flat-top ramp and plateau commands retain their
compiled playback order. Template grouping is not a public waveform
decomposition guarantee.

MuxReadout lowers the real compiled mixmux tones into simultaneous EventSlots
on one generator, with one static PFB ADC slot per declared tone. DDS, gain,
phase and demodulation rounding come from QICK's compiled parameters. Tone order
is ADC declaration order; channels are not sorted. The fixed mux compile profile
comes from caller-owned YAML through `program.build_soccfg`, with 1..8 explicit
ADC routes per mux block. Numeric environments
must separately bind every requested route and supply any physical mux response.

`ReadoutWindow` keeps ADC sample counts and integer sample periods independent
of generator duration. Per-point
windows retain trigger emission order, demodulation MHz/degrees and the generator
association for each read. The acquire-level windows specify fixed ADC shapes;
new consumers use template arrays for point-dependent timing/demodulation.
`materialize_point` expands one detached point for diagnostics, excluding reps.
Readout plans and environment templates convert integer tick grids to ns through
`TimeBase.ns`; frequency conversion to GHz remains with the physical consumers.
The request schema, compiled frontend and physical consumers use integer timing.
Physical planners receive the request's TimeBase, and plan/preparation cache keys
include it. Persisted acquire and round replay use format version 2, preserving
the rational time base and integer ticks. Older replay versions are rejected
without conversion.

ComputedPulse follows the selected compiled waveform and timestamp effects. The frontend does not pad a selected event to the longest
candidate. `add_table_sweep(name, values)` uses the same top-level axes as
`add_sweep(name, spec)`, with per-point wave fields or time-register values instead
of linear increments. Multiple tables on one loop keep their independent values.
The named loop's closing shell prepares the next values after every iteration,
including shots that do not play that waveform.
The last update restores table entry zero for the next outer loop or repetition.
Each point follows the corresponding scalar program's hardware words.
Linear/table parity additionally requires exact scalar re-encoding of the linear
sweep's actual floats. Truncating fields such as gain can differ by one least unit
on re-encoding; see [program table sweeps](../README.md#查表掃描).
Register and template state belongs to one frontend preparation and is not shared
with another acquire.
The binding retains detached immutable DAC envelopes in a 32 MiB LRU pool across
acquisitions. Every preparation checks current DAC content, normalization, clocks
and flat-top duration; names and program identities are not cross-acquire keys.
The budget includes keys, Python samples and containers. Eviction releases only
pool references, so existing requests keep their immutable envelopes unchanged.
Normal hardware compilation and the native timeline timing policy are unchanged.

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
  Without processors, threshold or trackers,
  it divides totals by reps and readout length, removes QICK offsets and moves reads first,
  without allocating `acc_buf`. Hooks receive the running mean of completed rounds.
  Every `acquire_summed` round materializes retained raw shots and uses the
  same joint-ADC processor and accumulation path as hardware. The processor
  receives ADC-keyed float64 `(points, reps, reads, 2)` normalized IQ without
  IQ-offset subtraction. It returns `(points, *axes)` sums; completed rounds
  add without normalization. Compact totals do not determine these sums.
  Herald and terminal reads retain paired outcome histories. Read-axis labels
  and classifiers follow the [program contract](../../README.md#summed-acquisition).
  Round hooks and the Schedule converter receive accumulated sums, not averages.
  Threshold, trackers and `get_raw()` materialize deterministic
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

`white_noise(soccfg, *, std_adc, seed, routes)` requires every input. It selects independent,
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
