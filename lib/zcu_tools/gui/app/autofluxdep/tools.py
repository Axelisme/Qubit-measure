"""Tools — sweep-lived stateful services curried into Nodes by their Builder.

The dependency model (``nodes/spec.py``) carries plain *values* between Nodes.
The **predictor** doesn't fit value-passing because it is stateful and shared
across the whole sweep:

- **predictor**: flux→qubit-freq prediction whose ``bias`` is *adapted* by
  qubit_freq's calibration after each successful point — state shared across
  Nodes AND across flux points. Its query face (``predict_freq``) is read by the
  predictor Service Node to produce ``predict_freq`` / ``cur_m``; its calibration
  face (``calibrate``) is a method a Node triggers, never the orchestrator.

The predictor lives in Tools (lifetime = the sweep) and is curried into Nodes via
the ``RunEnv.tools`` field. A Node never constructs it. The prototype binds a
``SimplePredictor`` stand-in; Phase B a real ``FluxoniumPredictor`` (scqubits).

``Smoother`` (pure smoothing mechanism) also lives here, but it is NOT curried
into Nodes — Nodes report raw values and never smooth. Smoothing is a
``SmoothingService`` (see ``autofluxdep.derivation``) that owns a ``Smoother``
and runs *after* the Nodes; ``Smoother`` is exported from here as the shared
mechanism it builds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional, Protocol

# SmoothMode is owned by nodes.spec (a dependency's ``smooth`` flag is one);
# re-used here so Smoother and the SmoothingService speak the same modes.
from zcu_tools.gui.app.autofluxdep.nodes.spec import SmoothMode


class Predictor(Protocol):
    """The stateful flux→freq predictor (FluxoniumPredictor in Phase B).

    *Query* face (pure): ``predict_freq`` / ``predict_matrix_element`` — used by
    the predictor Service Node to produce ``predict_freq`` / ``cur_m``.
    *Calibration* face (mutating, triggered by a Node not the orchestrator):
    ``calibrate`` folds a measured freq into the prediction so the next flux
    point's ``predict_freq`` reflects this point's measurement (adaptivity). The
    bias/IDW logic is encapsulated here; a Node never touches the internals.
    """

    def predict_freq(self, flux: float) -> float: ...
    def predict_matrix_element(self, flux: float) -> float: ...
    def calibrate(self, flux: float, measured_freq: float) -> None: ...


@dataclass
class SimplePredictor:
    """A prototype flux→freq stand-in (no scqubits, no FluxoniumPredictor).

    ``predict_freq(flux) = base + slope * flux`` — enough to move the synthetic
    Lorentzian across the sweep so the qubit_freq liveplot tracks a sloped line.
    The matrix element is a flat stand-in. ``calibrate`` nudges the base toward
    the measured freq (a crude adaptivity placeholder); Phase B swaps in the
    real ``FluxoniumPredictor`` (scqubits eigenvalues + bias root-find).
    """

    base: float = 5000.0
    slope: float = 50.0
    matrix_element: float = 0.1
    _bias: float = 0.0

    def predict_freq(self, flux: float) -> float:
        return self.base + self.slope * flux + self._bias

    def predict_matrix_element(self, flux: float) -> float:
        del flux
        return self.matrix_element

    def calibrate(self, flux: float, measured_freq: float) -> None:
        # nudge the bias so the next point's prediction drifts toward measured
        self._bias += 0.5 * (measured_freq - self.predict_freq(flux))


@dataclass
class Smoother:
    """Cross-point recursive smoothing, one history per named quantity.

    The pure mechanism a ``SmoothingService`` builds on (NOT called by Nodes —
    Nodes report raw). Two modes mirror the notebook:

    - ``ewma``:        ``0.5*(prev + cur)`` — t1 / t2r / t2e.
    - ``step_weighted``: ``w = decay**num_step; (1-w)*cur + w*prev`` where
      ``num_step`` is how many flux points since this quantity last updated —
      so a long gap (failed points) trusts the new value more. This is
      qubit_freq's qfw_factor and lenrabi's smooth_pi_product.

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

    def peek(self, name: str) -> Optional[float]:
        """The current smoothed estimate, or None if never updated."""
        return self._prev.get(name)


@dataclass
class Tools:
    """Container of the sweep-lived stateful services curried into Nodes.

    The controller builds this once per sweep (from the Setup resources) and the
    orchestrator curries it into every Node via ``RunEnv.tools``. ``predictor``
    is optional (the prototype binds a SimplePredictor; Phase B a real
    FluxoniumPredictor). Smoothing is deliberately NOT here — it is a
    DerivationService that runs after Nodes, not a tool Nodes call.
    """

    predictor: Optional[Predictor] = None
    # an IDW error-corrector / freq-error model joins here in Phase B, alongside
    # the real FluxoniumPredictor.
