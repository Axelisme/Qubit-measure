"""Tools — sweep-lived stateful services curried into Nodes by their Builder.

The dependency model (``nodes/spec.py``) carries plain *values* between Nodes.
Some run-lived capabilities do not fit value-passing because they are stateful
and shared across the whole sweep:

- **predictor**: flux→qubit-freq prediction whose ``bias`` is *adapted* by
  qubit_freq's calibration when the backend supports a physical bias update. Its
  query face is read by the predictor Service Node to produce base
  ``predict_freq`` / ``cur_m``; its calibration face is a method a Node triggers,
  never the orchestrator.
- **feedback**: placement-scoped scalar estimators/controllers whose state lives
  across flux points. Nodes decide what a correction/proposal means and when to
  observe/apply it.

Tools live for the sweep and are curried into Nodes via ``RunEnv.tools`` and
``RunEnv.feedback``. A Node never constructs them. The bound predictor is either
a ``SimplePredictor`` fallback or a real ``FluxoniumPredictor`` adapter.

``Smoother`` (pure smoothing mechanism) also lives here, but it is NOT curried
into Nodes — Nodes report raw values and never smooth. Smoothing is a
``SmoothingService`` (see ``autofluxdep.derivation``) that owns a ``Smoother``
and runs *after* the Nodes through the orchestrator/run-session path;
``Smoother`` is exported from here as the shared mechanism it builds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# SmoothMode is owned by nodes.spec (a dependency's ``smooth`` flag is one);
# re-used here so Smoother and the SmoothingService speak the same modes.
from zcu_tools.gui.app.autofluxdep.feedback import FeedbackRuntime
from zcu_tools.gui.app.autofluxdep.nodes.spec import SmoothMode


class Predictor(Protocol):
    """The stateful flux→freq predictor (SimplePredictor / FluxoniumPredictorAdapter).

    *Query* face (pure): ``predict_freq`` / ``predict_matrix_element`` — used by
    the predictor Service Node to produce ``predict_freq`` / ``cur_m``.
    *Calibration* face (mutating, triggered by a Node not the orchestrator):
    ``calibrate`` folds a measured freq into the physical/base predictor when the
    backend supports it. Generic residual correction lives in
    ``autofluxdep.feedback`` and is composed by the use-site node, not hidden here.
    """

    def predict_freq(self, flux: float) -> float: ...
    def predict_matrix_element(self, flux: float) -> float: ...
    def calibrate(self, flux: float, measured_freq: float) -> None: ...


@dataclass
class SimplePredictor:
    """A lightweight flux→freq fallback (no scqubits, no FluxoniumPredictor).

    Mirrors the real predictor's query shape, not its physics: a linear base
    (``base + slope * flux``) and a flat matrix element. It has no physical bias
    model, so ``calibrate`` is a no-op; residual correction is supplied by the
    generic feedback estimator owned by the use site.
    """

    base: float = 5000.0
    slope: float = 50.0
    matrix_element: float = 0.1

    def _physical(self, flux: float) -> float:
        return self.base + self.slope * flux

    def predict_freq(self, flux: float) -> float:
        return self._physical(flux)

    def predict_matrix_element(self, flux: float) -> float:
        del flux
        return self.matrix_element

    def calibrate(self, flux: float, measured_freq: float) -> None:
        del flux, measured_freq


@dataclass
class FluxoniumPredictorAdapter:
    """Wraps a real ``FluxoniumPredictor`` into the ``Predictor`` interface.

    The real predictor owns the physical model and its bias update. Residual
    interpolation is generic feedback state owned by the qubit_freq node's slot,
    so this adapter only exposes the base prediction and physical calibration.
    """

    fluxonium: Any  # a zcu_tools.simulate.fluxonium.FluxoniumPredictor

    def predict_freq(self, flux: float) -> float:
        return float(self.fluxonium.predict_freq(flux))

    def predict_matrix_element(self, flux: float) -> float:
        return float(self.fluxonium.predict_matrix_element(flux))

    def calibrate(self, flux: float, measured_freq: float) -> None:
        bias = self.fluxonium.calculate_bias(flux, measured_freq)
        self.fluxonium.update_bias(bias)


@dataclass
class Smoother:
    """Cross-point recursive smoothing, one history per named quantity.

    The pure mechanism a ``SmoothingService`` builds on (NOT called by Nodes —
    Nodes report raw). Two modes mirror the notebook:

    - ``ewma``:        ``0.5*(prev + cur)`` — t1 / t2r / t2e.
    - ``step_weighted``: ``w = decay**num_step; (1-w)*cur + w*prev`` where
      ``num_step`` is how many flux points since this quantity last updated —
      so a long gap (failed points) trusts the new value more. This mirrors the
      notebook's gap-aware tuning and is used by qubit_freq's ``qfw_factor``
      feedback.

    ``update(name, idx, cur, mode=...)`` folds ``cur`` into the running estimate
    and returns the smoothed value.
    """

    ewma_alpha: float = 0.5
    step_decay: float = 0.7

    _prev: dict[str, float] = field(default_factory=dict)
    _last_idx: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        name: str,
        idx: int,
        cur: float,
        mode: SmoothMode = "ewma",
    ) -> float:
        """Fold ``cur`` (this point's raw value at flux index ``idx``) into the
        running estimate for ``name`` and return the smoothed value."""
        prev = self._prev.get(name)
        if prev is None:
            smoothed = cur  # first observation: nothing to blend
        elif mode == "ewma":
            smoothed = self.ewma_alpha * prev + (1 - self.ewma_alpha) * cur
        else:  # step_weighted
            num_step = max(1, idx - self._last_idx.get(name, idx))
            weight = self.step_decay**num_step
            smoothed = (1 - weight) * cur + weight * prev
        self._prev[name] = smoothed
        self._last_idx[name] = idx
        return smoothed

    def peek(self, name: str) -> float | None:
        """The current smoothed estimate, or None if never updated."""
        return self._prev.get(name)


@dataclass
class Tools:
    """Container of the sweep-lived stateful services curried into Nodes.

    The controller builds this once per sweep (from the Setup resources) and the
    orchestrator curries it into every Node via ``RunEnv.tools``. ``predictor``
    is optional (SimplePredictor fallback or real FluxoniumPredictor adapter).
    Smoothing is deliberately NOT here: it runs after Nodes and is not a tool
    Nodes call.
    """

    predictor: Predictor | None = None
    feedback: FeedbackRuntime = field(default_factory=FeedbackRuntime)
