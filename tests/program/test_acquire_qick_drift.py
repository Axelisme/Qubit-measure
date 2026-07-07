"""Drift guard for the qick fork in EarlyStopMixin.

``EarlyStopMixin._finish_accumulated_round`` deliberately mirrors the
"accumulated" branch of ``qick.qick_asm.AcquireMixin.finish_round`` with
cancel_flag checks injected, because qick's loop has no early-exit seam
(its ``while`` only checks ``count < total_count``). The mirrored upstream
source is pinned here so a qick upgrade fails loudly instead of drifting
silently.

When this test fails after a qick upgrade:
1. Diff the snapshot against the new upstream source to see what changed.
2. Re-port the change into ``EarlyStopMixin._finish_accumulated_round``
   (lib/zcu_tools/program/base/improve_acquire.py).
3. Refresh the snapshot:
   .venv/bin/python -c "import inspect; from qick.qick_asm import AcquireMixin; \
open('tests/program/qick_finish_round.snapshot.txt', 'w').write(\
inspect.getsource(AcquireMixin.finish_round))"
"""

import inspect
from pathlib import Path

from qick.qick_asm import AcquireMixin

SNAPSHOT_PATH = Path(__file__).parent / "qick_finish_round.snapshot.txt"


def test_qick_finish_round_matches_mirrored_snapshot() -> None:
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8")
    actual = inspect.getsource(AcquireMixin.finish_round)
    assert actual == expected, (
        "qick's AcquireMixin.finish_round changed upstream. "
        "EarlyStopMixin._finish_accumulated_round mirrors its 'accumulated' "
        "branch and must be re-ported by hand; see this module's docstring "
        "for the procedure."
    )
