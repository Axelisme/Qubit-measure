"""Offline-only table publication and explicit bounded runtime loading."""

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from filelock import FileLock

from .archive import (
    ArchiveHeader,
    NumericArchive,
    encode_archive,
    read_archive,
    write_archive,
)
from .batch import BatchResult, CPUReference, PlateauFamily
from .controls import ControlSequence, quantize_ns
from .intel import IntelOpenCL
from .model import NumericModel
from .reuse import EvaluationReceipt
from .state import ComplexMatrix
from .validation import ValidationLimits, interpolate, validate_table


@dataclass(frozen=True)
class BuildPlan:
    """One explicit offline domain; never expanded/adapted at runtime.

    family: implemented joint plateau family. lower/upper: finite ordered scale
    interval. degree: polynomial interpolation degree, exactly 4 or 6. limits:
    absolute publication tolerances. A different domain/degree/limit has its own
    resumable plan identity; query scales do not change the family/table.
    """

    family: PlateauFamily
    lower: float
    upper: float
    degree: int
    limits: ValidationLimits = field(default_factory=ValidationLimits)

    def __post_init__(self) -> None:
        if (
            not np.isfinite([self.lower, self.upper]).all()
            or self.lower >= self.upper
            or self.degree not in (4, 6)
        ):
            raise ValueError("build requires finite ordered bounds and degree 4 or 6")

    @property
    def identity(self) -> str:
        """Identity of the domain, family and validation/numeric policy."""
        value = (
            self.family.identity,
            self.lower,
            self.upper,
            self.degree,
            asdict(self.limits),
        )
        return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class TableReference:
    """Immutable content reference: explicit path, file digest and matching domain.

    identity is the complete file SHA256, not merely a mutable path. family,
    lower/upper and degree are checked archive metadata. Deleting referenced
    files breaks reproduction; no implicit regeneration or disk eviction occurs.
    """

    path: Path
    identity: str
    family: str
    lower: float
    upper: float
    degree: int


@dataclass(frozen=True)
class LibrarySnapshot:
    """Fixed tuple of published versions, chosen by an acquire/round owner.

    tables: validated content references. New publications do not mutate this
    snapshot. Direct records are independent and immediately reusable.
    """

    tables: tuple[TableReference, ...]


@dataclass(frozen=True)
class BuildResult:
    """Offline progress and costs; a partial result is not an interpolation table.

    completed_nodes/computed_nodes: total retained/new nodes in this invocation.
    published: immutable reference only after all validation gates pass.
    batch: actual GPU batch costs, None when every node was already saved.
    validation_seconds: CPU comparison/heldouts/long-sequence/scale-fit time.
    file_bytes: published file size (zero for partial construction).
    """

    completed_nodes: int
    computed_nodes: int
    published: TableReference | None
    batch: BatchResult | None
    validation_seconds: float
    file_bytes: int


class EvolutionLibrary:
    """Explicit disk library and LRU for deterministic numeric arrays.

    path: caller-selected root, never cwd/global discovery. budget_bytes defaults
    to 4,000,000,000 and covers nodes, grid U arrays, completion masks and direct
    density arrays retained in this instance's host LRU. Runtime GPU residency
    is zero. Oversized entries are used transiently but not retained. Archive
    decoding/interpolation/build workspace, Python objects, models/solver/ADC and
    driver allocations are separate process memory, not hidden in this budget.
    last_evaluation: latest successful public Evolution receipt, or None before
    use. Contains no quantum/sample state; author diagnostic only.
    No disk eviction/daemon/background construction. Calls are synchronous;
    do not share an instance across concurrent computing threads. FileLock and
    atomic publication protect disk writes across processes.
    """

    def __init__(self, path: Path, *, budget_bytes: int = 4_000_000_000) -> None:
        if not path.is_absolute():
            raise ValueError("library requires an explicit absolute path")
        if isinstance(budget_bytes, bool) or budget_bytes <= 0:
            raise ValueError("resident array budget must be positive bytes")
        self.path = path
        self.budget_bytes = budget_bytes
        self._memory: OrderedDict[str, NumericArchive] = OrderedDict()
        self.last_evaluation: EvaluationReceipt | None = None

    @property
    def resident_bytes(self) -> int:
        """Current total host ndarray payload in the LRU; GPU residency is zero."""
        return sum(item.nbytes for item in self._memory.values())

    def clear_memory(self) -> None:
        """Evict all resident arrays, never disk data or sampled/quantum state."""
        self._memory.clear()

    def _load(self, path: Path, digest: str) -> NumericArchive:
        if digest in self._memory:
            self._memory.move_to_end(digest)
            return self._memory[digest]
        archive = read_archive(path, digest)
        if archive.nbytes <= self.budget_bytes:
            while (
                self._memory
                and self.resident_bytes + archive.nbytes > self.budget_bytes
            ):
                self._memory.popitem(last=False)
            self._memory[digest] = archive
        return archive

    def snapshot(self) -> LibrarySnapshot:
        """Capture/check published immutable files; missing root means no tables.

        Partial files are not discovered as publications. Corrupt published data
        raises, never becomes a miss. A snapshot holds references, not ndarrays.
        """
        tables = []
        for path in sorted((self.path / "tables").glob("*.npz")):
            archive = read_archive(path, path.stem)
            header = archive.header
            if header.kind != "table":
                raise ValueError("corrupt publication: expected complete table")
            tables.append(
                TableReference(
                    path,
                    path.stem,
                    header.family,
                    header.lower,
                    header.upper,
                    header.degree,
                )
            )
        return LibrarySnapshot(tuple(tables))

    def evaluate(
        self, family: PlateauFamily, scale: float, snapshot: LibrarySnapshot
    ) -> ComplexMatrix | None:
        """Interpolate a matching frozen table; None means only a legal miss.

        Reject nonfinite queries. Out-of-domain/different family is a miss, never
        extrapolation. Prefer higher fixed degree then content identity. This
        pure-U interface intentionally ignores baths; Evolution forbids applying
        the result to dissipative requests. Does not solve, save or publish.
        """
        if not np.isfinite(scale):
            raise ValueError("query scale must be finite")
        matches = [
            ref
            for ref in snapshot.tables
            if ref.family == family.identity and ref.lower <= scale <= ref.upper
        ]
        if not matches:
            return None
        ref = max(matches, key=lambda r: (r.degree, r.identity))
        archive = self._load(ref.path, ref.identity)
        return interpolate(archive.nodes, archive.values, scale)

    def read_record(self, key: str) -> tuple[ComplexMatrix, str] | None:
        """Read an exact deterministic density record, or None if absent.

        key is the library-generated exact request identity. Corruption/key
        mismatch fails explicitly; records are never promoted to U tables.
        """
        path = self.path / "direct" / f"{key}.npz"
        if not path.exists():
            return None
        archive = read_archive(path)
        if archive.header.kind != "direct" or archive.header.key != key:
            raise ValueError("direct record identity mismatch")
        return archive.values, sha256(path.read_bytes()).hexdigest()

    def save_record(self, key: str, density: ComplexMatrix) -> str:
        """Atomically persist a deterministic exact density; return file identity.

        No sampled populations/noise/filter state is retained. Disk write errors
        propagate to the acquire that incurred this direct solve.
        """
        path = self.path / "direct" / f"{key}.npz"
        archive = NumericArchive(
            ArchiveHeader(kind="direct", key=key),
            np.array([], dtype=np.float64),
            density,
            np.array([], dtype=np.bool_),
        )
        return write_archive(path, archive)

    def plateau(
        self,
        model: NumericModel,
        controls: ControlSequence,
        endpoints: tuple[float, float],
        snapshot: LibrarySnapshot,
    ) -> tuple[ComplexMatrix, str] | None:
        """Find one static closed joint interval in a frozen published family.

        Uses complete absolute phase history; only frequency=0 constant envelopes
        with full interval support qualify. Bath/dynamic requests are legal
        misses. Actual source amplitudes must be a real common scale of stored
        ratios. Joint H is never replaced by serial source propagators.
        """
        if (
            model.hamiltonian_at is not None
            or model.collapse_at is not None
            or model.collapse_operators
        ):
            return None
        start, end = endpoints
        active = [
            p
            for p in controls.pulses
            if quantize_ns(p.start_ns) <= start and quantize_ns(p.end_ns) >= end
        ]
        if any(p.frequency_ghz != 0 or p.envelope.constant is None for p in active):
            return None
        amplitudes = tuple(
            sum(
                (
                    controls.amplitude(p, (start + end) / 2)
                    for p in active
                    if p.source == source.name
                ),
                0j,
            )
            for source in model.sources
        )
        for ref in sorted(
            snapshot.tables, key=lambda r: (r.degree, r.identity), reverse=True
        ):
            archive = self._load(ref.path, ref.identity)
            header = archive.header
            if header.duration_ns != end - start or len(header.ratios) != len(
                amplitudes
            ):
                continue
            ratios = tuple(complex(real, imag) for real, imag in header.ratios)
            family = PlateauFamily(model, end - start, ratios)
            if family.identity != ref.family:
                continue
            nonzero = [i for i, ratio in enumerate(ratios) if ratio != 0]
            if not nonzero:
                continue
            quotient = amplitudes[nonzero[0]] / ratios[nonzero[0]]
            scale = quotient.real
            if abs(quotient.imag) > 1e-13 or any(
                abs(a - scale * r) > 1e-13
                for a, r in zip(amplitudes, ratios, strict=True)
            ):
                continue
            if header.lower <= scale <= header.upper:
                return interpolate(archive.nodes, archive.values, scale), ref.identity
        return None

    def build(
        self, plan: BuildPlan, backend: IntelOpenCL, *, node_limit: int | None = None
    ) -> BuildResult:
        """Compute/resume nodes with Intel OpenCL, validate, then atomically publish.

        node_limit optionally limits newly computed nodes for planned partial
        work; positive integer or None for all missing nodes. CPUReference is
        comparison only, never an alternate production builder. Validation,
        OpenCL, corrupt partial and write failures propagate. No order increase.
        Same-plan writers serialize on an explicit per-plan disk FileLock.
        """
        if node_limit is not None and (isinstance(node_limit, bool) or node_limit <= 0):
            raise ValueError("node_limit must be a positive integer")
        folder = self.path / "partial"
        folder.mkdir(parents=True, exist_ok=True)
        with FileLock(str(folder / f"{plan.identity}.lock")):
            return self._build_locked(plan, backend, node_limit)

    def _build_locked(
        self, plan: BuildPlan, backend: IntelOpenCL, node_limit: int | None
    ) -> BuildResult:
        path = self.path / "partial" / f"{plan.identity}.npz"
        if path.exists():
            partial = read_archive(path)
            if partial.header.key != plan.identity or partial.header.kind != "partial":
                raise ValueError("partial archive plan mismatch")
        else:
            nodes = (plan.lower + plan.upper) / 2 + (
                plan.upper - plan.lower
            ) / 2 * np.cos(np.linspace(np.pi, 0, plan.degree + 1))
            nodes[0], nodes[-1] = plan.lower, plan.upper
            n = len(plan.family.model.basis.labels)
            header = ArchiveHeader(
                kind="partial",
                key=plan.identity,
                family=plan.family.identity,
                lower=plan.lower,
                upper=plan.upper,
                degree=plan.degree,
                duration_ns=plan.family.duration_ns,
                ratios=tuple(
                    (complex(r).real, complex(r).imag) for r in plan.family.ratios
                ),
            )
            partial = NumericArchive(
                header,
                nodes,
                np.zeros((len(nodes), n, n), dtype=np.complex128),
                np.zeros(len(nodes), dtype=np.bool_),
            )
        indices = np.flatnonzero(~partial.present)[:node_limit]
        batch: BatchResult | None = None
        values, present = partial.values.copy(), partial.present.copy()
        if len(indices):
            batch = backend.compute(plan.family, partial.nodes[indices])
            values[indices] = batch.propagators
            present[indices] = True
        partial = NumericArchive(partial.header, partial.nodes, values, present)
        write_archive(path, partial)
        if not present.all():
            return BuildResult(int(present.sum()), len(indices), None, batch, 0, 0)
        start = perf_counter()
        reference = CPUReference().compute(plan.family, partial.nodes).propagators
        node_error = float(np.max(np.abs(reference - values)))
        if node_error > plan.limits.gpu_abs:
            raise ValueError(f"GPU node validation failed: {node_error}")
        evidence = validate_table(plan.family, partial.nodes, values, plan.limits)
        header = partial.header.model_copy(
            update={"kind": "table", "evidence": evidence + (("gpu", node_error),)}
        )
        table = NumericArchive(header, partial.nodes, values, present)
        digest = sha256(encode_archive(table)).hexdigest()
        destination = self.path / "tables" / f"{digest}.npz"
        write_archive(destination, table)
        ref = TableReference(
            destination,
            digest,
            plan.family.identity,
            plan.lower,
            plan.upper,
            plan.degree,
        )
        return BuildResult(
            len(present),
            len(indices),
            ref,
            batch,
            perf_counter() - start,
            destination.stat().st_size,
        )
