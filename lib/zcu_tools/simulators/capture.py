"""Project acquire observations into the debug capture manifest."""

from zcu_tools.program.v2.simulation import AcquireOutcome


def acquire_capture_metadata(
    outcome: AcquireOutcome, *, effective_settings: dict[str, object]
) -> dict[str, object]:
    """Return JSON metadata for one admitted acquire, including failed rounds.

    outcome supplies the binding's captured snapshots and normal-return flag.
    effective_settings is the simulator's JSON-mode settings after any drift.
    Sampling is null before the first successful snapshot; rounds list the
    attempted rounds in execution order with native DC values and model identity.
    This projection does not read live devices or repeat any sampling.
    """
    rounds = outcome.rounds
    request = rounds[0].acquire if rounds else None
    return {
        "complete": outcome.complete,
        "effective_settings": effective_settings,
        "sampling": (
            None
            if request is None
            else {"root_seed": request.root_seed, "sequence": request.sequence}
        ),
        "rounds": [
            {
                "index": round_request.index,
                "dc_values": dict(round_request.snapshot.values),
                "model_identity": round_request.snapshot.model_identity,
            }
            for round_request in rounds
        ],
    }
