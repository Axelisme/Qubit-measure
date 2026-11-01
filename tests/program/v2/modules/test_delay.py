import pytest
from zcu_tools.program.v2.modules.delay import DelayAuto


@pytest.mark.parametrize("t", [0.0, "cycles"])
def test_delay_auto_rejects_no_channels(t: float | str) -> None:
    with pytest.raises(ValueError, match="DelayAuto.*at least one.*channel"):
        DelayAuto("wait", t=t, gens=False, ros=False)


# ---------------------------------------------------------------------------
# DelayAuto — normal path (no tag, no reg_name)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Join — disable_delay called, runs interleaved
# ---------------------------------------------------------------------------
