"""Secular models and quantized flux in ns/GHz, independent of model builders."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from threading import Lock
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._validation import density, hermitian, integers, owned, ragged, shape

FluxLevel = tuple[int, ...]
"""One integer per loop: flux = round DC + level * 1e-3 Phi0.

The all-zero tuple denotes the DC model. Tuple order is supplied by the model
builder (F1, C, F2 for FTF).
"""


@dataclass(frozen=True, eq=False)
class SegmentShift:
    """Constant segment correction in the physical flux model's state order.

    start_ns, end_ns: finite absolute ns, 0 <= start < end. TimelineBatch
        requires an exact final segment span; adjacent spans may touch, not
        overlap. Missing spans are zero, not a hold of the previous shift.
    energy_shift_ghz: finite float64 (M,), additive energy/h in cycles/ns.
    dephasing_per_ns: nonnegative symmetric float64 (M,M), zero diagonal,
        additive coherence-pair decay in ns^-1, without another 2*pi.
    Arrays own immutable copies. No basis rotation or state reordering occurs.
    Shifted energies must stay nondecreasing (equal adjacent energies are valid).
    Engine.evolve rejects inversions with ValueError naming the point/segment:
    cavity-drive Stark shifts then exceed the dispersive model's validity.
    Shape must match the source model there. Invalid inputs raise
    ValueError before numerical work. Complete positivity is caller-owned.
    """

    start_ns: float
    end_ns: float
    energy_shift_ghz: NDArray[np.float64]
    dephasing_per_ns: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not (
            np.isfinite(self.start_ns)
            and np.isfinite(self.end_ns)
            and 0 <= self.start_ns < self.end_ns
        ):
            raise ValueError("shift span must be finite with 0 <= start < end")
        energy = owned(self.energy_shift_ghz, np.float64, "energy shift")
        if energy.ndim != 1 or not len(energy):
            raise ValueError("energy shift must have shape (states,)")
        gamma = owned(self.dephasing_per_ns, np.float64, "shift dephasing")
        shape(gamma, (len(energy), len(energy)), "shift dephasing")
        if np.any(gamma < 0) or np.any(np.diag(gamma) != 0):
            raise ValueError("shift dephasing must be nonnegative with zero diagonal")
        if not np.allclose(gamma, gamma.T, rtol=0, atol=1e-10):
            raise ValueError("shift dephasing must be symmetric")
        object.__setattr__(self, "energy_shift_ghz", energy)
        object.__setattr__(self, "dephasing_per_ns", gamma)

    @property
    def content(self) -> bytes:
        """Stable rate/energy content fingerprint; span is not model identity."""
        digest = sha256()
        digest.update(str(self.energy_shift_ghz.shape).encode())
        digest.update(self.energy_shift_ghz.tobytes())
        digest.update(self.dephasing_per_ns.tobytes())
        return digest.hexdigest().encode()


@dataclass(frozen=True)
class ModelIdentity:
    """Internal model identity: physical flux and independent shift content."""

    physical_flux: FluxLevel
    shift_content: bytes = b""


class ShiftOrderingError(ValueError):
    """A segment shift inverts the physical model state order."""


@dataclass(frozen=True)
class SecularDrive:
    """One port's Hermitian H/h quadratures in its model's eigenbasis.

    name: nonempty unique port name, matched by DriveTrack.port.
    in_phase_ghz, quadrature_ghz: complex128 (M,M), GHz per unit real/imaginary
        envelope; H_drive/h = Re(z)*X + Im(z)*Y, with no hidden 1/2.
    Arrays own read-only copies.

    Raises:
        ValueError: message names the invalid shape or Hermitian condition.
    """

    name: str
    in_phase_ghz: NDArray[np.complex128]
    quadrature_ghz: NDArray[np.complex128]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("drive requires a port name")
        x = hermitian(self.in_phase_ghz, "in-phase drive")
        y = hermitian(self.quadrature_ghz, "quadrature drive")
        shape(y, x.shape, "quadrature drive")
        object.__setattr__(self, "in_phase_ghz", x)
        object.__setattr__(self, "quadrature_ghz", y)


@dataclass(frozen=True)
class SecularModel:
    """One flux level's secular Lindblad data, in its own H0 eigenbasis.

    level: quantized flux, one integer per loop; zero is round DC.
    identity: nonempty content key including level and all model parameters;
        equal identities promise numerically identical data.
    energies_ghz: float64 (M,), ascending H0/h eigenvalues in cycles/ns.
    eigenvectors: complex128 (M,M) unitary columns in fixed DC coordinates;
        overlap from basis b to a is V_a.conj().T @ V_b.
    drives: uniquely named ports in this eigenbasis, all of dimension M.
    rates_per_ns: nonnegative float64 (M,M), R[target,source], zero diagonal;
        population generator is R - diag(R.sum(axis=0)).
    dephasing_per_ns: nonnegative symmetric float64 (M,M), zero diagonal;
        coherence decay is (R.sum(0)[i]+R.sum(0)[j])/2 + dephasing[i,j].
    Arrays own read-only copies. Shape, ordering, Hermiticity or unitary errors
    raise ValueError (matrix absolute tolerance 1e-10). Secular applicability
    and complete positivity of pure dephasing remain the builder's contract.

    Raises:
        ValueError: message names shape, finite data, ascending energies,
        unitary vectors, nonnegative rates, zero diagonal, or symmetric
        dephasing, according to the violated condition.
    """

    level: FluxLevel
    identity: str
    energies_ghz: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    drives: tuple[SecularDrive, ...]
    rates_per_ns: NDArray[np.float64]
    dephasing_per_ns: NDArray[np.float64]

    def __post_init__(self) -> None:
        energies = owned(self.energies_ghz, np.float64, "energies")
        if energies.ndim != 1 or not len(energies) or np.any(np.diff(energies) < 0):
            raise ValueError("energies must be a nonempty ascending vector")
        if not self.identity or not self.level:
            raise ValueError("model requires an identity and flux level")
        size = len(energies)
        vectors = owned(self.eigenvectors, np.complex128, "eigenvectors")
        shape(vectors, (size, size), "eigenvectors")
        if not np.allclose(
            vectors.conj().T @ vectors, np.eye(size), rtol=0, atol=1e-10
        ):
            raise ValueError("eigenvectors must be unitary")
        if len({drive.name for drive in self.drives}) != len(self.drives):
            raise ValueError("drive names must be unique")
        for drive in self.drives:
            shape(drive.in_phase_ghz, (size, size), "drive")
        rates = owned(self.rates_per_ns, np.float64, "rates")
        gamma = owned(self.dephasing_per_ns, np.float64, "dephasing")
        for name, matrix in (("rates", rates), ("dephasing", gamma)):
            shape(matrix, (size, size), name)
            if np.any(matrix < 0) or np.any(np.diag(matrix) != 0):
                raise ValueError(f"{name} must be nonnegative with zero diagonal")
        if not np.allclose(gamma, gamma.T, rtol=0, atol=1e-10):
            raise ValueError("dephasing must be symmetric")
        for name, value in (
            ("energies_ghz", energies),
            ("eigenvectors", vectors),
            ("rates_per_ns", rates),
            ("dephasing_per_ns", gamma),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FluxSchedule:
    """Per-point piecewise-constant flux, in batch point order.

    offsets: int64 (points+1,), strictly increasing from zero to total;
        point p occupies offsets[p]:offsets[p+1], at least one segment.
    start_ns: float64 (total,), strictly increasing within each point,
        beginning at 0. A segment extends until the next start or readout.
    level: int64 (total,loops), quantized FluxLevel of every segment;
        an inactive point has one segment at zero level.
    Arrays own read-only copies.

    Raises:
        ValueError: message names offsets for invalid spans, increase for
        unordered sample times, or shape for incompatible flux arrays.
    """

    offsets: NDArray[np.int64]
    start_ns: NDArray[np.float64]
    level: NDArray[np.int64]

    def __post_init__(self) -> None:
        offsets = integers(self.offsets, "flux offsets")
        starts = owned(self.start_ns, np.float64, "flux starts")
        levels = integers(self.level, "flux levels")
        ragged(offsets, starts)
        if levels.ndim != 2 or levels.shape[0] != len(starts) or levels.shape[1] == 0:
            raise ValueError("flux levels must have shape (total, loops)")
        if np.any(starts[offsets[:-1]] != 0):
            raise ValueError("first flux segment must start at zero")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "start_ns", starts)
        object.__setattr__(self, "level", levels)


@dataclass(frozen=True)
class FluxTerm:
    """One g(flux[loop])*matrix contribution in fixed DC coordinates.

    loop: nonnegative index into the builder's flux vector.
    kind: g is cos(2*pi*f), sin(2*pi*f), cos(pi*f), or sin(pi*f), respectively.
    matrix: Hermitian complex128 (M,M) H/h in GHz, owned/read-only.
    Raises:
        ValueError: message names invalid loop/function or Hermitian matrix.
    """

    loop: int
    kind: Literal["cos_2pi", "sin_2pi", "cos_pi", "sin_pi"]
    matrix: NDArray[np.complex128]

    def __post_init__(self) -> None:
        if self.loop < 0 or self.kind not in ("cos_2pi", "sin_2pi", "cos_pi", "sin_pi"):
            raise ValueError("invalid flux term loop or function")
        object.__setattr__(self, "matrix", hermitian(self.matrix, "flux term"))


@dataclass(frozen=True)
class FluxExpansion:
    """Exact H(flux)/h = static + sum(g_k(flux[loop_k])*terms[k].matrix).

    static: Hermitian complex128 (M,M) GHz in fixed DC coordinates,
        copied/read-only. terms: tuple of FluxTerm of the same dimension.
    Raises:
        ValueError: message names incompatible shape or Hermitian matrix.
    """

    static: NDArray[np.complex128]
    terms: tuple[FluxTerm, ...]

    def __post_init__(self) -> None:
        static = hermitian(self.static, "static flux Hamiltonian")
        for term in self.terms:
            shape(term.matrix, static.shape, "flux term")
        object.__setattr__(self, "static", static)


@dataclass(frozen=True)
class FluxWaveform:
    """Filtered flux samples on a DAC grid, ragged in batch point order.

    offsets: int64 (points+1,), increasing from zero to total, no empty point.
    time_ns: float64 (total,), nonnegative and increasing inside each point.
    flux: float64 (total,loops), absolute reduced flux in Phi0, not level
        offsets. Consumers interpolate linearly between samples.
    Arrays own read-only copies.

    Raises:
        ValueError: message names offsets for invalid spans, increase for
        unordered sample times, or shape for incompatible flux arrays.
    """

    offsets: NDArray[np.int64]
    time_ns: NDArray[np.float64]
    flux: NDArray[np.float64]

    def __post_init__(self) -> None:
        offsets = integers(self.offsets, "waveform offsets")
        times = owned(self.time_ns, np.float64, "waveform times")
        flux = owned(self.flux, np.float64, "waveform flux")
        ragged(offsets, times)
        if flux.ndim != 2 or flux.shape[0] != len(times) or flux.shape[1] == 0:
            raise ValueError("waveform flux must have shape (total, loops)")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "time_ns", times)
        object.__setattr__(self, "flux", flux)


class ModelSource:
    """Round-local, thread-safe model cache owned by the numerical engine API.

    initial_density: complex128 (M,M) Gibbs density in the DC eigenbasis;
        owned/read-only, Hermitian and positive with trace one (atol 1e-8).
    build: pure callable supplied by the model package; returns a SecularModel
        for each requested level. It owns physics/device-limit validation.
    expansion: optional exact fixed-coordinate flux Hamiltonian expansion.
    slow_envelope_segments_per_period: strict positive integer subdivisions per
        carrier period for midpoint plateau propagation, default 20. Immutable
        numerical policy; invalid values raise ValueError at construction.
    Construction rejects invalid density with ValueError. No density is passed
    to build; preparing a kernel must not inspect initial_density.
    """

    def __init__(
        self,
        initial_density: NDArray[np.complex128],
        build: Callable[[FluxLevel], SecularModel],
        expansion: FluxExpansion | None = None,
        *,
        slow_envelope_segments_per_period: int = 20,
    ) -> None:
        if (
            type(slow_envelope_segments_per_period) is not int
            or slow_envelope_segments_per_period < 1
        ):
            raise ValueError(
                "slow_envelope segments per period must be a positive integer"
            )
        self._slow_envelope_segments_per_period = slow_envelope_segments_per_period
        self._initial_density = density(initial_density)
        self._build = build
        self._expansion = expansion
        self._models: dict[FluxLevel, SecularModel] = {}
        self._shift_models: dict[ModelIdentity, SecularModel] = {}
        self._shifts: dict[bytes, SegmentShift] = {}
        self._gate_usage: dict[bytes, int] = {}
        self._gate_contents: dict[bytes, set[bytes]] = {}
        self._lock = Lock()

    @property
    def initial_density(self) -> NDArray[np.complex128]:
        """Read-only DC Gibbs density, complex128 (M,M) in its eigenbasis."""
        return self._initial_density

    @property
    def slow_envelope_segments_per_period(self) -> int:
        """Read-only positive midpoint subdivisions per carrier period, default 20."""
        return self._slow_envelope_segments_per_period

    def model(self, level: FluxLevel) -> SecularModel:
        """Build once per level, serializing cache access and builds across threads.

        Builder errors propagate unchanged and are not cached. The builder must
        not recursively call this same source while holding its build lock.
        """
        with self._lock:
            if level not in self._models:
                self._models[level] = self._build(level)
            return self._models[level]

    def register_shift(self, shift: SegmentShift) -> bytes:
        """Register a correction and return its span-independent content bytes.

        shift: validated SegmentShift with additive energy/h in GHz, shape (M,),
            and coherence-pair dephasing in ns^-1, shape (M,M), in the physical
            model's state order. Compatibility with a model is checked only
            when shifted_model resolves it; registration does not build models.

        Returns shift.content, fingerprinting the correction arrays, not their
        start/end span. Equal content keeps the first registration and any
        cached shifted models. Even zero correction arrays have nonempty
        content; b'' is reserved by shifted_model for the unshifted model.
        Registration and model caches persist until close(), which clears them.
        Register again before resolving nonempty content after close().
        """
        content = shift.content
        with self._lock:
            self._shifts.setdefault(content, shift)
        return content

    def shifted_model(self, level: FluxLevel, content: bytes = b"") -> SecularModel:
        """Return the physical model with a registered energy/dephasing correction.

        level: physical FluxLevel in the builder's loop order, with one integer
            per loop; flux = round DC + level * 1e-3 Phi0. The all-zero tuple
            selects the DC model.
        content: bytes returned by register_shift on this source since its last
            close(). The default b'' returns model(level) unchanged and needs no
            registration. Nonempty content must be registered, even if all
            correction values are zero. The registered span is ignored here.

        Adds the registered GHz energies and ns^-1 coherence-pair dephasing to
        model(level) and includes content in the returned model's identity.
        Physical level, eigenvectors, drive quadratures, transition rates and
        initial_density stay unchanged; no state reordering or basis rotation
        occurs. Successful shifted models are cached per (level, content), with
        serialized cache access. close() clears this cache and registrations.

        Builder errors propagate unchanged, before content lookup. Unregistered
        nonempty content raises KeyError. Incompatible correction/model shapes
        or invalid resulting SecularModel data raise ValueError. Energy-order
        inversion raises ValueError indicating that the cavity-drive Stark
        shift exceeds dispersive validity; equal adjacent energies are valid.
        Failed shifted models are not cached; the physical model may be cached.
        """
        model = self.model(level)
        if not content:
            return model
        identity = ModelIdentity(level, content)
        with self._lock:
            if identity not in self._shift_models:
                shift = self._shifts[content]
                shape(shift.energy_shift_ghz, model.energies_ghz.shape, "energy shift")
                energies = model.energies_ghz + shift.energy_shift_ghz
                if np.any(np.diff(energies) < 0):
                    raise ShiftOrderingError(
                        "cavity-drive Stark shift reverses energy ordering; "
                        "outside dispersive model validity"
                    )
                self._shift_models[identity] = replace(
                    model,
                    identity=model.identity + ":shift:" + content.decode(),
                    energies_ghz=energies,
                    dephasing_per_ns=model.dephasing_per_ns + shift.dephasing_per_ns,
                )
            return self._shift_models[identity]

    @property
    def gate_usage(self) -> dict[bytes, int]:
        """Snapshot of distinct acquire content's complete row inventories.

        Each content contributes its canonical gate/multitone row multiplicities
        once, including rejected and nonresident keys. Identical repeats need no
        numerical playback, so counting them would overprice the Sambe/RWA
        alternative and select unnecessary gate builds. Counts and content
        identities persist across cache eviction/Engine replacement until close.
        """
        with self._lock:
            return self._gate_usage.copy()

    def projected_gate_usage(
        self, inventory: Mapping[bytes, int], *, content: bytes
    ) -> dict[bytes, int]:
        """Return totals after a hypothetical acquire, without changing history.

        Only the supplied physics keys are visited. Calls sharing this source
        serialize projection and the following record/evolve operation.
        """
        with self._lock:
            seen = self._gate_contents.get(content, set())
            return {
                key: self._gate_usage.get(key, 0) + (0 if key in seen else count)
                for key, count in inventory.items()
            }

    def record_gate_usage(
        self, inventory: Mapping[bytes, int], *, content: bytes
    ) -> dict[bytes, int]:
        """Count distinct acquire content's complete row inventories.

        content is evolution's complete numerical-input digest, independent of
        cache state. Each namespaced physics key contributes its nonnegative row
        multiplicity only on its first appearance in that content. Repeated
        contents return the current totals without incrementing them, whether
        preparation/result caches hit, are cleared, evict or are disabled.
        History is source-owned and unbounded, like the cumulative key counters;
        close releases both. Invalid counts raise ValueError before mutation.
        Callers serialize acquires sharing a source, even across Engines.
        """
        if any(count < 0 for count in inventory.values()):
            raise ValueError("gate usage counts must be nonnegative")
        with self._lock:
            seen = self._gate_contents.setdefault(content, set())
            for key, count in inventory.items():
                if key not in seen:
                    self._gate_usage[key] = self._gate_usage.get(key, 0) + count
                    seen.add(key)
            return {key: self._gate_usage[key] for key in inventory}

    def close(self) -> None:
        """Release model caches and cumulative usage; idempotent, reusable.

        Immutable builder/initial inputs remain valid. A later evolve starts a
        new usage history. Call only after all work sharing this source drains.
        """
        with self._lock:
            self._models.clear()
            self._shift_models.clear()
            self._shifts.clear()
            self._gate_usage.clear()
            self._gate_contents.clear()

    def flux_expansion(self) -> FluxExpansion:
        """Return the exact supplied expansion, or raise ValueError if absent."""
        if self._expansion is None:
            raise ValueError("model source has no flux expansion")
        return self._expansion
