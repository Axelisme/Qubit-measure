# Explicit acquisition environments

**Last updated:** 2026-10-19，binding-owned envelope reuse

`SimulationAcquireMixin` enters once through `MyProgramV2`. With a
`SimulationSoc`, it admits a synchronous call, builds one `AcquireRequest`,
and supplies numeric round data to the existing QICK acquisition pipeline.
`SimulationBinding` optionally accepts a `wait_ready` callback. Admission calls
it before consuming the acquire sequence or constructing a request. Failure
releases admission and propagates; the environment callback owns any optional
startup-compilation fallback. External timers must wait at the environment's
ready interface before starting if startup time should be excluded.
Other SoC objects use the original path. Classification, hooks, tracker,
normalization, averaging and Stop still belong to that pipeline.

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

The measure launcher accepts `--simulate /absolute/path/noise_environment.py`.
It imports the file and calls `build_environment()` without arguments once.
No environment path means normal mode, without an automatic board connection.
Load or installation errors abort startup without a fallback.

Notebook callers import the same builder and use its returned objects:

```python
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
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
        env.soccfg, ProgramV2Cfg(reps=4, rounds=2),
        modules=[readout.build("readout")],
    )
    averaged = prog.acquire(env.soc, progress=False)
    raw = prog.get_raw()
finally:
    env.devices.close_all_devices()
```

`SimulationEnvironment` returns `soc`, matching `soccfg`, `devices` and a public
`description`. The builder owns cleanup before successful return. The host owns
cleanup after return, even if publication or window assembly fails. GUI injects
the same `DeviceManager`, publishes live drivers through `DeviceService`, and
restores preferences afterward. Same-name live devices win; other remembered
rows remain offline. After operation admission closes and work drains, composition
closes the manager. Failed driver closes remain registered and propagate. There
is no automatic retry. `Controller.close()` drains work but does not close drivers.

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
uses the covariance of the sum over reps. Raw materialization uses the retained
round identity, so requesting shots does not rerun evolution or draw a new round.
Raw ADC values remain int64, and round indices select distinct sampling streams.

Python loop counts do not grow with shots or sweep points. Template construction,
grouping and sampling use NumPy arrays or numba kernels for point-dependent work.
Python iterates over structural templates and unique content, not individual
points or reps.

## Numeric contract and extension points

`core.simulation` owns immutable numeric records. No Module, driver, GUI object
or callback crosses this boundary. Time is us, frequency MHz, phase degrees and
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
  memory. `execute(RoundRequest)` returns `RoundStatistics` for integrated mode
  or `RoundSignals` for decimated mode. Mode/result mismatches raise.
- Integrated statistics contain per-point totals over reps, shared state counts,
  conditional per-shot sums and receiver covariance. The adapter validates and
  detaches these arrays. Without threshold, classification or trackers, it divides
  totals by reps and readout length, removes QICK offsets and moves reads first,
  without allocating `acc_buf`. Hooks receive the running mean of completed rounds.
  Threshold, classification, trackers and `get_raw()` materialize deterministic
  int64 shots from the retained last-round identity. Getters never rerun a backend.
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
admission on normal return or exception. Stop retains the ordinary completed-round
or no-result behavior; it does not interrupt a synchronous backend immediately.

Only `SimulationProvenance.identifier` is copied to public Run metadata as optional
`simulation_id`. Backend/RNG versions and private requests remain on the typed
binding. Seeds, device-round inputs and model content are not put in normal GUI
or MCP metadata. Persistent private replay is a separate delivery.

General program classes depend on the public mixin entry, not frontend, binding,
signal-packing or SoC internals. `.importlinter` enforces that direct dependency
boundary while allowing the entry to delegate internally. `experiment.simulation`
composes program and device capabilities. Numeric backends depend on core records,
not on program or GUI owners.
