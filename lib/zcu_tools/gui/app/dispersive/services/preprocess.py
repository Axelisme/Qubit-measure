"""PreprocessService — the one-tone signal-preprocessing pipeline (notebook cells 5-6).

Extracts the normalized phase image the dispersive tuning / fit work against from
the raw complex S-parameter signals: fit + remove the electronic delay, smooth,
fit a common circle centre, take the phase, then differentiate / abs / row-normalize.

Pure and Qt-free. ``fast_edelays`` first discovers one electrical-delay branch shared
by every flux row, then the analysis numba kernel JIT-compiles the per-row 1000-point
local circle refinement and parallelises the flux loop in ``prange``. numba releases
the GIL, so the parallelism needs no process fork — nothing is pickled, so a Qt
``GuiProgressBar`` cannot leak across a worker boundary; since the kernel is a single
black-box call (about 0.1 s for a representative 64 × 301 warm run), the GUI shows a
busy/indeterminate bar rather than per-flux ticks. The whole ``compute`` still runs on
a worker thread so it cannot block the event loop. The remaining steps reuse the
resonance primitives from ``zcu_tools.analysis.fitting.resonance``.
"""

from __future__ import annotations

from zcu_tools.analysis.dispersive.models import PreprocessResult
from zcu_tools.analysis.dispersive.preprocess import compute_preprocess
from zcu_tools.gui.app.dispersive.state import DispersiveState


class PreprocessService:
    """Runs the preprocessing pipeline on the loaded one-tone, writes the result."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def compute(self) -> PreprocessResult:
        """Run the pipeline on the loaded one-tone — pure, off-main-safe (no State write).

        Snapshots the spectrum off State first (a fast read), then runs the heavy
        pipeline. Pair with ``record`` on the main thread. Fast-fails when no
        one-tone is loaded.
        """
        entry = self._state.onetone
        if entry is None:
            raise RuntimeError("no one-tone spectrum loaded (call load_onetone first)")
        raw = entry.raw
        return compute_preprocess(raw["fluxs"], raw["freqs"], raw["signals"])

    def record(self, result: PreprocessResult) -> None:
        """Write a computed preprocessing result onto State (MAIN THREAD only)."""
        self._state.set_preprocess(result)

    def preprocess(self) -> PreprocessResult:
        """Compute + record inline (RPC / convenience path, main thread)."""
        result = self.compute()
        self.record(result)
        return result
