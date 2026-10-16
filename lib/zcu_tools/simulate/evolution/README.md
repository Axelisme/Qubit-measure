# Segment evolution and prepared-data reuse

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
flux waveform varies during a driven segment, `flux_dop853` interpolates
`TimelineBatch.flux_waveform` and evaluates the exact `FluxExpansion` in its
RHS, including tails that stay inside one grid level. The model owner truncates
tails at its absolute threshold and holds the terminal DC sample. The bath stays
at the row's grid level and switches at segment boundaries.
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

## Default execution and replay

FTF acquisition uses this engine's fast segment algorithms by default. Kernel
selection and preparation happen inside the engine, not in the caller. The
prepare cache holds deterministic numeric payloads in memory. It is not an
offline U-table library, and evolution does not persist exact density records.
There is no database builder or GPU dependency.

`experiment.ftf` owns private replay inputs and composes evolution with readout.
Replay recomputes a round from its archived model, templates, snapshot and
sampling identity. It requires matching `SEGMENT_NUMERIC_VERSION`, backend,
model, RNG and dependency versions. No record references pin old densities or
offline tables. Older exact-record manifests fail explicitly.

The prepare-cache budget charges prepared payloads, not process RSS. Models,
returned densities, live chunk references, solver workspaces and ADC preparations
are separate allocations. Clearing the cache or closing the engine releases
in-memory preparations without deleting replay inputs.
