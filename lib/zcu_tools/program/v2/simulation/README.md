# Explicit acquisition environments

`SimulationAcquireMixin` enters once through `MyProgramV2`. With a
`SimulationSoc`, it admits a synchronous call, builds one `AcquireRequest`,
and supplies numeric round data to the existing QICK acquisition pipeline.
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

## Numeric contract and extension points

`core.simulation` owns immutable numeric records. No Module, driver, GUI object
or callback crosses this boundary. Time is us, frequency MHz, phase degrees and
gain normalized DAC amplitude. The initial frontend accepts exact scalar
`PulseReadout` modules with constant waveforms and a single `freq` for generator
and ADC matching. It rejects other Modules, runtime controls and waveforms rather
than approximating them. Full Module interpretation extends `frontend.py`.

- `AcquireRequest` fixes C-order `loop_dims`, `avg_level`, per-execution controls,
  declared ADC order, trigger-order windows and private sampling identity.
- `EnvironmentSnapshot` contains numeric native-unit device values and retained
  model/calibration identity. The environment captures it immediately before
  each round, not in getters or preparation. A successful round hook may change
  the next snapshot. It cannot change this acquire's controls or layout.
- `SimulationBinding(capture_snapshot, execute, backend_version=..., seed=...)`
  coordinates admission and retains `last_request` and `rounds` privately in
  memory. `execute(RoundRequest)` returns all ADCs in one `RoundSignals`.
- Integrated IQ has shape `(execution, read, 2)` and contains float64 ADC sums.
  Decimated IQ has shape `(execution, read, sample, 2)` and contains float64 ADC
  samples. Last axis is I/Q. Backend arrays are borrowed until the adapter copies
  them. Missing, duplicate or extra ADCs, wrong shapes and nonfinite values fail.
- The adapter orders channels, sums companion raw from the same decimated trace,
  and applies nearest-even int64 rounding. Overflow fails instead of wrapping or
  clipping. QICK divides integrated sums by readout length and averages only at
  its existing levels. Decimated acquisition averages software rounds, not reps.

`white_noise(std_adc=1.0, seed=None, ...)` explicitly selects independent,
zero-mean Gaussian I/Q per channel/sample. `std_adc` is finite, nonnegative and
measured per ADC sample. Integrated sums use standard deviation
`std_adc * sqrt(samples)`. Zero produces exact zero. It models no pulse response.
The root seed is supplied or generated once. PCG64 streams derive from
`SeedSequence([root_seed, acquire_sequence, round_index])`; NumPy and derivation
versions are retained. Calls, including failed or stopped calls, consume distinct
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
