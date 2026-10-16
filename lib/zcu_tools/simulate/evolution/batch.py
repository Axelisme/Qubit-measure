"""The implemented fixed-DC joint plateau family and CPU reference."""

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .model import NumericModel, hermitian_matrix
from .state import ComplexMatrix


@dataclass(frozen=True)
class PlateauFamily:
    """One joint constant-H family H(scale)=H0+scale*drive, in GHz.

    model: fixed-DC numeric model, including all simultaneous source operators.
        Dynamic Hamiltonian callbacks are unsupported. Baths do not enter U or
        its identity; runtime must not use U for a dissipative request.
    duration_ns: positive nearest-2-ns duration, not independently rounded here.
    ratios: one finite signed complex amplitude per model.sources entry. Phase
        is included in each ratio. All channels act together, not serially.
    identity hashes actual H0, basis coordinates/order, operators, ratios,
    duration and numerical convention, never output paths or bath parameters.
    """

    model: NumericModel
    duration_ns: float
    ratios: tuple[complex, ...]

    def __post_init__(self) -> None:
        if self.model.hamiltonian_at is not None:
            raise ValueError("plateau family requires a static Hamiltonian")
        if (
            not np.isfinite(self.duration_ns)
            or self.duration_ns <= 0
            or self.duration_ns % 2
        ):
            raise ValueError("plateau duration must be positive on the 2 ns grid")
        if (
            len(self.ratios) != len(self.model.sources)
            or not np.isfinite(self.ratios).all()
        ):
            raise ValueError("ratios must match sources and be finite")

    @property
    def drive_ghz(self) -> ComplexMatrix:
        """Return the combined Hermitian drive per dimensionless scale."""
        drive = np.zeros_like(self.model.hamiltonian_ghz)
        for source, ratio in zip(self.model.sources, self.ratios, strict=True):
            drive += ratio.real * source.in_phase_ghz
            if source.quadrature_ghz is not None:
                drive += ratio.imag * source.quadrature_ghz
        return hermitian_matrix(drive, "joint plateau drive")

    @property
    def identity(self) -> str:
        """Reproducible SHA256 of the family, independent of query scale/baths."""
        digest = sha256(b"joint-plateau-complex128-v1")
        basis = self.model.basis
        for value in (
            basis.identity,
            *basis.labels,
            *(s.name for s in self.model.sources),
        ):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        for array in (
            basis.vectors,
            self.model.hamiltonian_ghz,
            np.array([self.duration_ns]),
            np.array(self.ratios),
        ):
            digest.update(np.asarray(array, dtype="<c16").tobytes())
        for source in self.model.sources:
            digest.update(np.asarray(source.in_phase_ghz, dtype="<c16").tobytes())
            digest.update(np.asarray(source.quadrature_ghz, dtype="<c16").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class BatchResult:
    """Batch outputs and measured workspace costs, not a process RAM limit.

    propagators: owned complex128 (scale,state,state) matrices.
    device: actual computing implementation/device name. compute_seconds:
        wall time of CPU expm or GPU kernel; transfer_seconds: both GPU transfers.
    compile_seconds: OpenCL build, zero for CPU. host_bytes: explicit host input
        and output ndarrays. gpu_bytes: explicit device buffers plus conservative
        per-work-item private matrix storage (driver allocations are additional).
    """

    propagators: NDArray[np.complex128]
    device: str
    compute_seconds: float
    transfer_seconds: float = 0.0
    compile_seconds: float = 0.0
    host_bytes: int = 0
    gpu_bytes: int = 0


def batch_hamiltonians(
    family: PlateauFamily, scales: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Validate a finite nonempty scale vector and form joint H/h in GHz."""
    scales = np.asarray(scales, dtype=np.float64)
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all():
        raise ValueError("batch scales must be a nonempty finite vector")
    return family.model.hamiltonian_ghz[None] + scales[:, None, None] * family.drive_ghz


class CPUReference:
    """Independent scipy complex128 exponential reference; not a production builder."""

    def compute(
        self, family: PlateauFamily, scales: NDArray[np.float64]
    ) -> BatchResult:
        """Return exp(-2πi H(scale)*duration_ns) for each finite scale."""
        h = batch_hamiltonians(family, scales)
        start = perf_counter()
        result = np.array(
            [expm(-2j * np.pi * family.duration_ns * item) for item in h],
            dtype=np.complex128,
        )
        return BatchResult(
            result,
            "CPU scipy expm",
            perf_counter() - start,
            host_bytes=h.nbytes + result.nbytes,
        )
