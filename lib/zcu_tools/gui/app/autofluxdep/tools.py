"""Tools — orchestrator-owned, sweep-lived stateful services injected into Nodes.

The Node dependency model (``nodes/spec.py``) carries plain *values* between
Nodes. The **predictor** doesn't fit value-passing because it is stateful and
shared across the whole sweep:

- **predictor** (FluxoniumPredictor): flux→qubit-freq prediction whose ``bias``
  is *adapted* by qubit_freq after each successful point — state shared across
  Nodes AND across flux points. Read in build_cfg (predict_freq) and the
  per-point pre-step; written in result post-processing (update_bias).

The predictor is owned by the Orchestrator (lifetime = the sweep) and injected
into Nodes via the ``tools`` parameter. A Node never constructs it.

``Smoother`` (pure smoothing mechanism) also lives here, but it is NOT a tool
injected into Nodes — Nodes report raw values and never smooth. Smoothing is a
``SmoothingService`` (see ``autofluxdep.derivation``) that owns a ``Smoother``
and runs *after* the Nodes; ``Smoother`` is exported from here as the shared
mechanism it builds on.

The real ``FluxoniumPredictor`` is bound in Phase B; here predictor is a
Protocol so the model is expressible without importing the simulate stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Optional, Protocol

# SmoothMode is owned by nodes.spec (a dependency's ``smooth`` flag is one);
# re-used here so Smoother and the SmoothingService speak the same modes.
from zcu_tools.gui.app.autofluxdep.nodes.spec import SmoothMode


class Predictor(Protocol):
    """The stateful flux→freq predictor (FluxoniumPredictor in Phase B).

    Reads are pure; ``update_bias`` mutates internal state so the next flux
    point's ``predict_freq`` reflects this point's measurement (adaptivity).
    """

    def predict_freq(self, flux: float) -> float: ...
    def predict_matrix_element(self, flux: float) -> float: ...
    def calculate_bias(self, flux: float, measured_freq: float) -> Any: ...
    def update_bias(self, bias: Any) -> None: ...


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
    """Container of the sweep-lived stateful services injected into Nodes.

    Orchestrator builds this once per sweep and passes it to every Node's
    build_cfg / post-processing. ``predictor`` is optional in the skeleton
    (Phase B binds the real one). Smoothing is deliberately NOT here — it is a
    DerivationService that runs after Nodes, not a tool Nodes call.
    """

    predictor: Optional[Predictor] = None
    # freq_err_pred (the qubit_freq frequency-error model) is a second predictor
    # added in Phase B alongside the real FluxoniumPredictor.
