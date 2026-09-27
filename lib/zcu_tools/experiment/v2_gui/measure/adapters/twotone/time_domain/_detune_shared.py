"""Shared detune-ratio plumbing for the T2 Ramsey / T2 Echo adapters.

Both adapters carry a run-only ``detune_ratio`` knob (fringes-per-step, not part
of their experiment Cfg): it is stripped before lowering, and converted to the
absolute applied detune (MHz) as ``detune_ratio / sweep_step``. These three
pieces are identical between the two adapters; the per-adapter differences (the
Cfg type, Ramsey's ``true_detune`` stash + ``q_f`` writeback) stay in each
adapter.
"""

from __future__ import annotations


def detune_ratio_of(raw_cfg: dict[str, object]) -> float:
    """Read and validate the run-only ``detune_ratio`` knob from the raw cfg."""
    value = raw_cfg.get("detune_ratio")
    if not isinstance(value, (int, float)):
        raise ValueError("detune_ratio must be a number")
    return float(value)


def strip_detune_ratio(raw_cfg: dict[str, object]) -> dict[str, object]:
    """Return a copy of ``raw_cfg`` without the run-only ``detune_ratio`` key.

    The experiment Cfg has no ``detune_ratio`` field and rejects unknown keys, so
    it must be stripped before ``ml.make_cfg`` (mirrors ro_optimize/auto's
    num_points)."""
    cfg_raw = dict(raw_cfg)
    cfg_raw.pop("detune_ratio", None)
    return cfg_raw


def resolve_detune(detune_ratio: float, sweep_step: float) -> float:
    """Convert the fringes-per-step ratio to the absolute applied detune (MHz).

    The absolute detune is ``detune_ratio / sweep_step``, where ``sweep_step`` is
    the lowered length sweep step (us). Mirrors the notebook's
    ``activate_detune = ratio / cfg.sweep.length.step``. A zero ratio means no
    fringe (detune 0), skipping the divide; a nonzero ratio on a degenerate sweep
    (step == 0, expts < 2) Fast-Fails rather than dividing by zero.
    """
    if detune_ratio == 0.0:
        return 0.0
    if sweep_step == 0.0:
        raise ValueError(
            "cannot apply a nonzero detune_ratio on a degenerate "
            "length sweep (expts < 2, step == 0)"
        )
    return detune_ratio / sweep_step
