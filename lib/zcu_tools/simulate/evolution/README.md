# Evolution APIs and data reuse

**Last updated:** 2026-10-17

## Segment engine contracts

The batch API uses ns, GHz and rad. `SecularModel` carries energy eigenvalues,
drive quadratures, directional rates and pure dephasing at one quantized flux
level. `ModelSource` caches each level under a lock. The model builder owns
secular applicability and device limits. Engine records own read-only array
copies and reject invalid shapes and numeric invariants at construction.

`EnvelopeSpec` describes unity or sampled complex amplitudes. Its shared
`values` method interpolates real and imaginary parts and holds the final
sample. `TimelineBatch` supplies all points of a template. `EvolutionResult`
represents densities at readout start in the DC eigenbasis. `Engine` segments
all templates with array event sorting, selects kernels by name, and runs
independent joblib-threading chunks. Missing required registry entries raise
with the segment and kernel name. T4 implements exact zero-drive secular
propagation, including quantized flux subsegments with basis changes. When a
flux grid level changes during a drive, `flux_dop853` interpolates the optional
`TimelineBatch.flux_waveform` and evaluates the exact `FluxExpansion` in its
RHS. The bath stays at the row's grid level and switches at segment boundaries.
Waveform variation that stays within one grid level keeps the quantized path.
Other algorithm families plug into the internal registry.

The engine counts prepare keys with `np.unique`. Repeated keys are prepared
in the shared A stage. Single-use keys are prepared inside their timeline
chunk and discarded afterward. The LRU charges whole opaque prepared batches;
a live acquire can retain evicted payloads until its chunks finish. The cache
namespace includes the kernel name, zero-level model identity and budget.
The zero-level identity must describe the entire source, including all model
parameters and absolute DC flux. Row keys also include quantized flux levels.

Chunk-local waves call apply once per group of next rows sharing a kernel and
prepared batch. Consecutive DOP853 rows form chains only within the same kernel
and prepared batch, never across `dop853` and `flux_dop853`. Their kernels own
basis changes inside a chain; the engine changes basis between apply calls
and returns the final state to the DC eigenbasis. Cancellation is checked at
chunk boundaries and raises `Cancelled` without partial results. `close()`
waits for active work, clears the prepare cache and is safe to repeat. Each
`evolve` call owns its threading pool; no worker resources survive the call.
`SEGMENT_NUMERIC_VERSION` identifies engine numerics for checked replay.

The internal `kernel` module defines `SegmentTable`, `Kernel` and the opaque
`Prepared.nbytes` contract. Only apply receives density arrays. The shared
contract suite lives in `tests/simulate/evolution/contract/kernel_suite.py`.
Kernel tests inherit `KernelContract` and provide a `kernel_case` fixture with
accepted rows, a synthetic model and input densities. The suite compares against
an independent strict DOP853 reference and checks row splitting and preparation
without density access. The demo kernel stays in tests.

## Existing Evolution API

`Evolution` owns event geometry, solver selection, U lookup and exact density
records. The caller supplies a model, continuous controls and an explicit initial
density. No shot, filter tail or noise realization survives a call implicitly.
`quantize_ns` exposes the same absolute nearest-2-ns, ties-even event geometry
used by controls. Composition callers use it when anchoring a local carrier to
an external common phase reference; local reset/continuous semantics do not change.
The solver contracts exact Lindblad jump algebra before repeated RHS evaluations.
Dynamic callbacks still run and validate their full contents at every evaluation.
Only identical contents reuse algebra. Hamiltonian and baths stay in one full
Lindblad ODE, without operator splitting or fewer bath updates. Each action uses
at most 16 MiB for an optional dense gain matrix; dimensions above 32 use exact
batched matrix products instead. Solver workspaces remain outside the library
LRU budget. `NUMERIC_VERSION` is `density-direct-v2`; older exact-record keys and
private replay versions do not silently alias the changed arithmetic.
`last_evaluation` reports the actual direct, direct-record or table path and
content identities. It is author diagnostic data, not public acquisition truth.

## Offline construction

`PlateauFamily` describes fixed DC, constant-envelope, zero-frequency joint
controls with fixed signed complex source ratios. `BuildPlan` selects one common
amplitude interval and polynomial degree 4 or 6. Duration, phase, source operators,
ratios, Hamiltonian and basis coordinates participate in family identity. A query
inside the amplitude interval does not create another table. Baths do not enter
U identity, but runtime never applies these U tables to dissipative requests.

`IntelOpenCL(compiler_directory)` computes complex128 batches on the local Intel
Arc OpenCL device. The compiler directory is explicit. `CPUReference` uses SciPy
expm as an independent comparison, not a production fallback. Other GPU vendors
have no implementation or empty registration entry.

`EvolutionLibrary.build` saves completed nodes under `partial/` and resumes the
same plan. Partial files cannot be evaluated as tables. Before publication the
builder compares GPU nodes to CPU reference, then tests independent heldouts,
phase-bearing densities, adjacent blocks, 512 repeats, independent trace and
unitarity limits, and recovered
amplitude-scale calibration. It publishes only when every absolute limit passes.
`BuildResult.validation` and `library.validation(reference)` expose named
absolute error measurements without requiring callers to decode files. Partial
results have no validation evidence. Serialization/publication cost is separate
from numerical validation. The default limits are numerical checks for this family, not chip calibration or
basis convergence guarantees. A failed degree does not trigger a higher degree.

## Persistence and runtime

The caller must inject an absolute library path. No cwd or global cache discovery
occurs. Version 1 files contain a checked JSON header and non-pickle NumPy arrays,
with an internal SHA256. Published table filenames are complete-file SHA256
identities. Writes use a same-directory temporary file, fsync and atomic replace.
A per-plan FileLock serializes partial builders. Disk data is never auto-deleted.

`library.snapshot()` fixes published content references at an author-selected
round or acquire boundary. The snapshot contains no arrays. New publications do
not change it. `evaluate(family, scale, snapshot)` returns U or a legal miss,
without extrapolation or automatic building. Missing/corrupt referenced files,
illegal input, solver and write errors propagate.

`replay_manifest(snapshot, content_identities)` returns an opaque checked JSON
manifest for the original table set and exact record content used by a round.
`restore_replay(manifest)` checks every referenced file from disk and returns
that frozen table set with pinned exact-record references, without discovering
new publications or warming arrays. Replay ignores later exact records that
were not part of the original evaluation. A listed exact record must still match
its original file digest at lookup; an unlisted record cannot override a U table.
The numeric owner resolves record keys and checks versions, digests and metadata.
Caller archives retain this opaque string, not an independent NPZ parser. Files
remain in the explicitly selected library; deleting them breaks replay rather
than triggering silent regeneration.

Pass `library=` and optionally `tables=` to `Evolution`. Static closed plateau
intervals can use the snapshot. All other legal requests keep the full direct
solver, including dynamic flux, arbitrary continuous envelopes and Lindblad
baths. A direct result saves one deterministic density record, not a table node
or a sampled signal. It can be reused immediately and across processes.

Exact keys include initial density, basis, Hamiltonian/source matrices, collapse
operators, controls and phase history, callback-presence semantics, numerical
settings and library versions. Built-in envelope constructors provide content
identities. Custom envelopes need a stable `identity`; models with callbacks need
`dynamics_identity` describing every callback dependency. Callers must keep these
callbacks deterministic and immutable. Callable repr and whole YAML text are not
numeric identities. Seeds, shots, readout noise and output paths are not inputs
to these deterministic evolution records.

## Memory accounting

Each library instance defaults to 4,000,000,000 resident host-array bytes. Its LRU
counts loaded grid U arrays, scale nodes and completion masks. Direct records
currently stream through without residency. An oversized table also streams
through. Eviction and `clear_memory()` do not delete files or alter results.
Runtime GPU residency is zero.

The limit is not a process RAM cap. Archive decoding, returned U/density matrices,
models, solver/bath workspaces, ADC preparations and Python objects are outside
it. `BatchResult` separately reports build host H/output arrays, device buffers
and a conservative private-kernel matrix allowance. Driver allocations and
measured process RSS must be reported separately. On an integrated GPU, host and
device allocations share physical RAM; their reservation totals are not two
independent physical-memory pools.
