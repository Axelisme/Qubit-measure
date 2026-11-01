import pytest
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, Join, SoftDelay
from zcu_tools.program.v2.modules.pulse import Pulse










@pytest.mark.parametrize("t", [0.0, "cycles"])
def test_delay_auto_rejects_no_channels(t: float | str) -> None:
    with pytest.raises(ValueError, match="DelayAuto.*at least one.*channel"):
        DelayAuto("wait", t=t, gens=False, ros=False)


def test_delay_auto_with_tag_rejects_reg_name():
    with pytest.raises(
        ValueError, match="DelayAuto with tag cannot have t as a register name"
    ):
        DelayAuto("d", t="reg_name", tag="k")








# ---------------------------------------------------------------------------
# DelayAuto — normal path (no tag, no reg_name)
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# Join — disable_delay called, runs interleaved
# ---------------------------------------------------------------------------




