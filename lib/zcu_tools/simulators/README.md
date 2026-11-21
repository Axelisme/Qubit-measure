# Physical simulator models

**Last updated:** 2026-11-21, raw materialization timing

**Readers:** framework-dev, lab-author, agent. Locate physical model configuration, numerics and environment builders.

[ADR-0072](../../../docs/adr/0072-module-boundaries.md) defines model placement and framework dependencies.
[ADR-0078](../../../docs/adr/0078-evolution-module-boundary.md) governs evolution integration and preparation lifetimes.
[ADR-0084](../../../docs/adr/0084-simulation-integer-time-base.md) defines integer request times.

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
boundary. `AcquireTelemetry` requires the first eleven fields below.
`RoutingRecord` inherits them and adds the four writer-generated fields.

| Field | Meaning |
| --- | --- |
| `acquire_id` | Opaque identity for matching deferred work to this acquire. |
| `points` | Request grid size, excluding shots and rounds. |
| `rounds` | Captured attempts, including failed attempts. |
| `wall_s` | Admitted acquire wall seconds, including pacing and excluding telemetry I/O. |
| `compile_wall_s` | Compilation wall seconds from the binding's frozen `AcquireOutcome`. |
| `evolve_wall_s` | Sum of every `Engine.evolve` call's wall seconds across rounds and continuations. |
| `readout_wall_s` | Sum of ADC/readout wall seconds, including preparation and integrated or decimated sampling. |
| `materialize_wall_s` | Wall seconds spent generating and detaching raw shots within the acquire, from `AcquireOutcome`. No raw materialization contributes `0.0`. |
| `complete` | `true` when the acquire body and cleanup returned normally, including a partial Stop result. `false` when an exception escaped. |
| `extras` | Simulator-prefixed keys with JSON values, or `{}` when absent. |
| `capture_file` | Capture basename, or `null` when recording is off or publication fails. |
| `timestamp` | Writer-generated UTC ISO timestamp. |
| `frontend_cpu_s` | Geometry and work-plan calling-thread CPU seconds. |
| `estimated_cost_unit` | `relative scheduling units`, the label for family scheduling costs. |
| `families` | Engine family names mapped to merged routing counters, CPU seconds and optional OpenCL detection error. |

All five wall fields are floats. A failed call retains elapsed time through its
exception; a stage not entered contributes `0.0`. `complete` comes from the binding
and does not mean every requested round ran. FTF measures cavity-memory spans and
projective continuations. Single and six also include projective prefixes and
continuations. Unmeasured parts of `wall_s` include environment preparation,
pacing and host shot processing.

When a later getter successfully materializes retained raw shots, the same
writer appends a separate JSONL record:

```json
{"event":"materialize","acquire_id":"opaque-acquire-id","materialize_wall_s":0.25,"timestamp":"2026-11-21T10:15:00+00:00"}
```

Match it to the original acquire by `acquire_id`. The supplemental duration
is outside that acquire's `wall_s`. It does not count as another acquire and
cannot change the original row or completion status. A failed materialization
or an access to cached shots appends nothing. The same retention budget
applies to both record kinds; the matching acquire row can be absent if pruned.
Older acquire rows may have neither `acquire_id` nor `materialize_wall_s`.

Use extras keys such as `ftf.projection_outcomes`; the prefix names the simulator's
extension, not a row identity. Required extras keys are indexed directly by their
consumer. The writer projects only declared fields and never modifies the caller's
summary. It stores no numerical inputs or simulator identity.
Each family entry contains `rows`, `duration_ns`, `prepare_cpu_s`, `apply_cpu_s`,
`actual_cpu_s`, `cache_hits`, `rejected_rows`, `estimated_cost`, `estimated_cpu_s`,
`prepare_builds`, `admission_rows`, `opencl_rows` and `opencl_detection_error`.
These fields use the [Engine report's aggregation rules](../simulate/evolution/README.md#routing-telemetry).
The writer lists the fields explicitly so new Engine diagnostics do not add JSONL keys.
Acquire logs have no per-row method field and do not store the Engine's `dp54_rows`
or `dop853_rows` endpoint counts. Family CPU excludes ADC, sampling and routing
bookkeeping. It is not wall time.

`numerics.telemetry.enabled` defaults to true in all three environments.
Their default `numerics.telemetry.directory=null` selects the same project-root
`.sim_telemetry` directory. To analyze environments separately, set a different
`telemetry.directory` under each environment's `numerics`, then run the analyzer
once per directory. Passing multiple directories to one invocation merges them
into one report. No path, filename, family or backend version identifies a row's
simulator.

Files use UTC dates in `routing-YYYYMMDD.jsonl`. Before appending, the writer deletes
oldest owned files by modification time, breaking ties by filename, until the new
line fits within the 50 MiB directory budget. It may remove and recreate today's
file. It leaves unrelated files alone and does not write a line larger than the
budget. Each writer instance serializes its writes. Filesystem or serialization
failure emits at most one `RuntimeWarning` per instance and never changes the
acquire result, even when warnings are configured as errors.

The [routing analyzer](../../../tools/README.md#routing-telemetry-analysis) reports
wall totals under `normal_return`, `raised` and `unknown`, using only `complete`.
Missing or null `complete` is unknown. Missing or null wall fields are unmeasured.
Only rows with numeric `wall_s` and all three numeric stages contribute wall totals
and the remainder, including unknown rows. Each observed group reports `included`
and `excluded`; their sum across groups equals `records`. Absent groups mean zero
rows. Every valid row still contributes family statistics. Non-null `complete`
must be a JSON boolean. Non-null wall fields must be JSON numbers, not booleans or
numeric strings. Wrong types skip the entire row, including its family counters,
and produce a file/line warning and a `skipped` count.

## Debug capture

Set `numerics.debug_capture: true` in FTF, single or six settings to record
Engine inputs and actual routes. The default is false. Six's top-level switch
owns one buffer for all six sources; nested qubit switches do not publish files.
One admitted acquire creates one buffer and passes it to every evolve.
After return or failure, on_end publishes one manifest before building telemetry.
A compile failure publishes a manifest with no Engine inputs. Readiness or on_begin failure
does not call on_end and leaves no capture. A later acquire failure does not
change the complete flag of an earlier successful evolve.

Recording works with `numerics.telemetry.enabled: false` and writes no JSONL row
in that mode. Otherwise the row's `capture_file` contains the successful
manifest basename. Off and failed publication both produce null. Older rows
without the key have no associated capture; the routing analyzer does not use it.
The binding has already measured `wall_s` before either writer runs.

Use `zcu_tools.simulate.evolution.capture_directory()` to locate files.
It returns `<user-cache-root>/simulator-debug/v1`, shared across worktrees and
tasks. `ZCU_TOOLS_CACHE_DIR` overrides the shared root with a non-empty absolute
path. The [capture writer](../simulate/evolution/README.md#capture-persistence)
defines atomic publication, codec failures and retention.

Open the gzip JSON manifest. Its `inputs` list holds frozen Engine inputs,
actual families and candidate decisions. The [capture inspector](../../../tools/README.md#simulator-capture-inspection)
summarizes those routes and matches the telemetry row by `capture_file`.
FTF and single also write a top-level `metadata` object:

- `rounds`: attempted rounds in order, including failed execution after a
  successful snapshot. Each entry has its zero-based `index`, `dc_values`
  by source name in the units declared in the settings, and `model_identity`.
- `effective_settings`: the settings captured for each observed round
  `model_identity`. Direct FTF callers can supply detached round snapshots. If
  the capture did not observe a round's identity, `rounds` still lists the round
  but `effective_settings` has no entry for it. Drift-enabled settings contain
  the sampled losses and `drift: null`. Changes after the final round do not appear.
- `admission_settings`: the effective settings at admission if the outcome has
  no round snapshots, such as after a compile failure. In that case
  `effective_settings` is empty. Otherwise `admission_settings` is null.
- `sampling`: the actual `root_seed` and acquire `sequence` from the request.
  A null configured seed still produces a root seed. `sampling` is null
  if the outcome has no round snapshots.
- `complete`: acquire return status, distinct from each evolve's status.
  A normal partial Stop is true; an escaping exception is false.

For FTF projective continuation, [FTF capture metadata](ftf/environment/README.md#debug-capture)
links Engine rows to original points, spans and histories.
Captures contain numerical inputs and model settings. Keep them out of ordinary Run metadata and Git.

## Test owners

`tests/simulators/<model>/config/` owns YAML validation.
`tests/simulators/<model>/environment/` owns public planning and acquisition behavior.
`tests/simulators/ftf/physics/` owns joint-model physics and its simulation marker.
`tests/simulators/test_telemetry.py` owns shared summary serialization, file rotation
and warning isolation. `tests/simulators/test_capture.py` owns the shared capture
lifecycle through public acquire with non-evolving Engine collaborators.
Writer/codec publication belongs to `tests/simulate/evolution/test_capture_writer.py`.
FTF telemetry integration stays with the FTF environment tests.
Generic engine, ADC, bath and source-line tests stay with their framework owners.
