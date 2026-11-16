# Physical simulator models

**Last updated:** 2026-11-16

**Readers:** framework-dev, lab-author, agent. Locate physical model configuration, numerics and environment builders.

[ADR-0072](../../../docs/adr/0072-module-boundaries.md) defines model placement and framework dependencies.
[ADR-0078](../../../docs/adr/0078-evolution-module-boundary.md) governs evolution integration and preparation lifetimes.
[ADR-0084](../../../docs/adr/0084-simulation-integer-time-base.md) defines integer request times and replay v2.

`simulators` composes framework mechanisms into three physical environments.
Each model owns its topology, YAML schema and environment assembly.
Import the builder from its `environment` package and the YAML loader from `config`:

```python
from zcu_tools.simulators.ftf.config import load_ftf
from zcu_tools.simulators.ftf.environment import ftf

settings = load_ftf(model_path)
environment = ftf(settings, soccfg=soccfg)
```

The caller supplies `model_path` and a `QickConfig` built from independent board YAML.
Model YAML explicitly supplies RNG seeds, bath occupations, pruning thresholds and
existing numerical controls. A null root seed chooses once at startup; a null
bath occupation uses the circuit temperature. Every receiver supplies background
period and pivot, even when zero depth and slope disable the effect.
Off/identity defaults remain optional. Missing required fields report dotted paths.
After acquisitions drain, the host closes the backend and then its devices.
The builders retain the framework's `SimulationEnvironment` and public acquisition contracts.

| Model | Configuration | Environment | Model numerics |
| --- | --- | --- | --- |
| FTF | [config](ftf/config/README.md) | [environment](ftf/environment/README.md) | [physics](ftf/physics/README.md) |
| Single fluxonium | [config](single_fluxonium/config/README.md) | [environment](single_fluxonium/environment/README.md) | Reuses `simulate.fluxonium` |
| Six fluxonium | [config](six_fluxonium/config/README.md) | [environment](six_fluxonium/environment/README.md) | Composes six independent single sources |

Model roots do not load environments. Configuration can be imported without loading runtime composition.
There are no compatibility imports at the former `experiment`, `resources` or `simulate.ftf` locations.

## Framework boundary

`simulate` retains evolution, ADC sampling, baths, dephasing, numeric drift,
source-line response and component Hamiltonians. `experiment` retains environment
bundles, `white_noise`, readout planning and common drive-track conversion.
`resources.simulation_schema` and `resources.yaml` retain neutral records and YAML decoding.
The `framework-no-simulators` forbidden import contract covers every framework package.
`simulators` may consume those mechanisms; framework packages cannot import model code.
C9 also checks top-level import cycles. C16 prevents models from importing lab definitions.

Startup scripts and caller-owned model YAML live in
[`zcu_lab/simulation_envs`](../../../zcu_lab/simulation_envs/README.md).
The startup changes only select the new model imports.

## Acquire telemetry

`simulators.telemetry` owns `RoutingTelemetry`, `TelemetrySettings` and the shared
acquire-summary types. FTF, single and six use this writer. Each admitted acquire
writes one row, including numeric failures. Six aggregates all sources and routed
groups before writing; individual sources do not write rows.

Call `RoutingTelemetry(directory).write(summary, paths=paths)` once at the acquire
boundary. `summary: AcquireTelemetry` requires `points`, `rounds`, `wall_s`,
`compile_wall_s`, `evolve_wall_s`, `readout_wall_s`, `complete` and `extras`.
Times are wall seconds; the three stages accept `None` when not measured.
Each environment projects `compile_wall_s` and `complete` from the binding's frozen
`AcquireOutcome`. Completion means normal return, including a partial Stop result,
not completion of every requested round. Single and six fill all stage values
with floats. Their evolve totals include projective prefixes and continuations.
ADC preparation, integrated sampling and decimated sampling contribute to readout.
Failed calls retain elapsed stage time; stages not entered remain `0.0`.
Deferred raw materialization after acquire is outside these totals.
FTF's evolve/readout slots remain `None` until its stage instrumentation is installed.
Use simulator-prefixed extras keys and JSON values, or `{}` when absent.
The writer projects only the declared fields and does not modify the caller's
summary. `RoutingRecord` inherits them and adds a UTC timestamp, frontend CPU
seconds, the cost-unit label and aggregated Engine family counters.
No numerical inputs enter the record.

The writer retains the `routing-*.jsonl` filenames, 50 MiB directory budget,
per-instance lock and warning-once I/O policy described in the
[FTF telemetry contract](ftf/environment/README.md#routing-telemetry).

## Private replay

`simulators.replay` owns opt-in private FTF and single-fluxonium input JSON,
content integrity, atomic file I/O and acquire/round reference resolution.
It retains detached waveform samples and settings, not measured I/Q.
Template arrays preserve dtype, shape and flat values, with typed float64, int64
and bool reconstruction. Acquire and round files use version 2 and reject version 1.
Replay requests require explicit `reps` and `relax_delay_ticks`; complete version 2
archives already contain both fields. Request clocks retain positive integer
num/den fields in MHz.
All time arrays retain int64 ticks, including ADC and envelope periods.
Numeric manifests remain opaque and are verified by their numeric owner.
Replay does not change normal waveform assets or data.h5 policies.

## Test owners

`tests/simulators/<model>/config/` owns YAML validation.
`tests/simulators/<model>/environment/` owns public planning and acquisition behavior.
`tests/simulators/ftf/physics/` owns joint-model physics and its simulation marker.
`tests/simulators/test_replay.py` owns private archive roundtrips and integrity errors.
`tests/simulators/test_telemetry.py` owns shared summary serialization, file rotation
and warning isolation. FTF acquire integration stays with the FTF environment tests.
Generic engine, ADC, bath and source-line tests stay with their framework owners.
