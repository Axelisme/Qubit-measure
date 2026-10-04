"""Framework-owned archive representations consumed by IO, exports and remote views.

Protocols expose read-only attributes containing mutable float64 arrays.
Concrete experiments and loaded archive records satisfy the same representation.
The literal result_kind describes the internal representation, not HDF5 metadata.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, TypedDict, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class FrequencySweepResult(Protocol):
    """Frequency sweep arrays, with NaN for unmeasured or unfitted values.

    Every array has dtype float64. Consumers may read and mutate array elements,
    but cannot replace arrays through this interface. The IO owner checks shapes.
    result_kind is the class-level archive declaration "qubit_freq".
    """

    result_kind: ClassVar[Literal["qubit_freq"]]

    @property
    def flux(self) -> NDArray[np.float64]:
        """Device coordinates, shape (n_flux,)."""
        ...

    @property
    def detune(self) -> NDArray[np.float64]:
        """MHz detuning coordinates, shape (n_detune,)."""
        ...

    @property
    def signal(self) -> NDArray[np.float64]:
        """Real signal, shape (n_flux, n_detune)."""
        ...

    @property
    def fit_curve(self) -> NDArray[np.float64]:
        """Fitted real signal, shape (n_flux, n_detune)."""
        ...

    @property
    def fit_freq(self) -> NDArray[np.float64]:
        """Absolute fitted frequency in MHz, shape (n_flux,)."""
        ...

    @property
    def predict_freq(self) -> NDArray[np.float64]:
        """Predicted frequency in MHz, shape (n_flux,)."""
        ...

    @property
    def snr(self) -> NDArray[np.float64]:
        """Signal-to-noise ratio, shape (n_flux,)."""
        ...


@runtime_checkable
class SweepResult1D(Protocol):
    """One-dimensional sweep arrays, with NaN for unmeasured or unfitted values.

    Every array has dtype float64. Consumers may read and mutate array elements,
    but cannot replace arrays through this interface. The IO owner checks shapes.
    result_kind is the class-level archive declaration "sweep1d".
    """

    result_kind: ClassVar[Literal["sweep1d"]]

    @property
    def flux(self) -> NDArray[np.float64]:
        """Device coordinates, shape (n_flux,)."""
        ...

    @property
    def x(self) -> NDArray[np.float64]:
        """Trailing coordinates, shape (n_x,), with meaning given by x_label."""
        ...

    @property
    def signal(self) -> NDArray[np.float64]:
        """Real signal, shape (n_flux, n_x)."""
        ...

    @property
    def fit_curve(self) -> NDArray[np.float64]:
        """Fitted real signal, shape (n_flux, n_x)."""
        ...

    @property
    def fit_value(self) -> NDArray[np.float64]:
        """Primary fitted scalar, shape (n_flux,), in the experiment's unit."""
        ...

    @property
    def snr(self) -> NDArray[np.float64]:
        """Signal-to-noise ratio, shape (n_flux,)."""
        ...

    @property
    def x_label(self) -> str:
        """Display label for x coordinates; the experiment defines its unit."""
        ...


@runtime_checkable
class SweepResult2D(Protocol):
    """Two-dimensional sweep arrays, with NaN for unmeasured or unselected values.

    Every array has dtype float64. Consumers may read and mutate array elements,
    but cannot replace arrays through this interface. The IO owner checks shapes.
    result_kind is the class-level archive declaration "sweep2d".
    """

    result_kind: ClassVar[Literal["sweep2d"]]

    @property
    def flux(self) -> NDArray[np.float64]:
        """Device coordinates, shape (n_flux,)."""
        ...

    @property
    def freq(self) -> NDArray[np.float64]:
        """Frequency coordinates in MHz, shape (n_freq,)."""
        ...

    @property
    def gain(self) -> NDArray[np.float64]:
        """Gain coordinates in the experiment's unit, shape (n_gain,)."""
        ...

    @property
    def signal(self) -> NDArray[np.float64]:
        """Real signal, shape (n_flux, n_freq, n_gain)."""
        ...

    @property
    def best_freq(self) -> NDArray[np.float64]:
        """Selected frequency in MHz, shape (n_flux,)."""
        ...

    @property
    def best_gain(self) -> NDArray[np.float64]:
        """Selected gain in the experiment's unit, shape (n_flux,)."""
        ...


WorkflowResult = FrequencySweepResult | SweepResult1D | SweepResult2D


class FrequencySweepFitSummary(TypedDict):
    """n_fitted counts finite fit_freq rows; last_fit_freq is MHz or None."""

    n_fitted: int
    last_fit_freq: float | None


class Sweep1DFitSummary(TypedDict):
    """n_fitted counts finite fit_value rows; last_fit_value uses the node unit.

    x_label identifies the sweep axis. None means no finite fit_value is present.
    """

    n_fitted: int
    last_fit_value: float | None
    x_label: str


class Sweep2DFitSummary(TypedDict):
    """n_fitted counts finite best_freq rows; last best values may be None.

    last_best_freq uses MHz; last_best_gain uses the experiment's gain unit.
    Each None means that scalar has no finite value.
    """

    n_fitted: int
    last_best_freq: float | None
    last_best_gain: float | None


class FrequencySweepProgressSummary(TypedDict):
    """qubit_freq progress: n_flux total, n_measured finite raw-signal rows."""

    kind: Literal["qubit_freq"]
    n_flux: int
    n_measured: int
    fit_summary: FrequencySweepFitSummary


class Sweep1DProgressSummary(TypedDict):
    """sweep1d progress: n_flux total, n_measured finite raw-signal rows."""

    kind: Literal["sweep1d"]
    n_flux: int
    n_measured: int
    fit_summary: Sweep1DFitSummary


class Sweep2DProgressSummary(TypedDict):
    """sweep2d progress: n_flux total, n_measured finite raw-signal rows."""

    kind: Literal["sweep2d"]
    n_flux: int
    n_measured: int
    fit_summary: Sweep2DFitSummary


ResultProgressSummary = (
    FrequencySweepProgressSummary | Sweep1DProgressSummary | Sweep2DProgressSummary
)


class FrequencySweepRowSummary(TypedDict):
    """One row: fit_freq/predict_freq in MHz, snr unitless; non-finite is None."""

    fit_freq: float | None
    predict_freq: float | None
    snr: float | None


class Sweep1DRowSummary(TypedDict):
    """One row: fit_value in the node's unit and snr; non-finite is None."""

    fit_value: float | None
    snr: float | None


class Sweep2DRowSummary(TypedDict):
    """One row: best_freq in MHz, best_gain in the node unit; non-finite is None."""

    best_freq: float | None
    best_gain: float | None


ResultRowSummary = FrequencySweepRowSummary | Sweep1DRowSummary | Sweep2DRowSummary


def require_workflow_result(result: object) -> WorkflowResult:
    """Return a result after checking its literal kind, Protocol, dtype and shapes.

    Accept an instance declaring qubit_freq, sweep1d or sweep2d at class level.
    Arrays must be float64 and match the axis sizes documented by its Protocol.
    NaN and infinite scalar values remain data, not validation failures.
    No arrays or attributes are mutated. Raise TypeError for missing/unknown kind,
    missing fields or invalid dtypes; raise ValueError for inconsistent shapes.
    IO and export owners share this check; class-only declaration lookup does not
    use it because a class need not have allocated arrays.
    """
    del result  # Orchestrator declaration seed; the assigned writer fills validation.
    raise NotImplementedError("result validation implementation is not prepared")
