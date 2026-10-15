# Evolution data reuse

**Last updated:** 2026-10-16

`Evolution` owns event geometry, solver selection, U lookup and exact density
records. The caller supplies a model, continuous controls and an explicit initial
density. No shot, filter tail or noise realization survives a call implicitly.
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
phase-bearing densities, adjacent blocks, 512 repeats, unitarity and recovered
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
