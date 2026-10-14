"""Scientific detune-ratio conversion shared by Ramsey and Echo Forms."""


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
