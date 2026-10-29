"""Actual waveform/ADC ends govern module emission at the public compiler seam."""

from collections.abc import Callable, Sequence

import pytest
from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules import Module, Pulse, PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program.v2.timing_support import shots, waves


def _pulse_cfg(
    ch: int = 0,
    *,
    length: float | QickParam = 0.101,
    pre: float | QickParam = 0.01,
    post: float | QickParam = 0.0005,
) -> PulseCfg:
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=1000,
        gain=0.5,
        waveform=ConstWaveformCfg(length=length),
        pre_delay=pre,
        post_delay=post,
    )


def _program(
    modules: Sequence[Module],
    body: Callable[[TimelineProgramV2], None],
    initialize: Callable[[TimelineProgramV2], None] | None = None,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            if initialize is not None:
                initialize(self)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            body(self)

    return Program(make_mock_soccfg(3, 1), reps=2, initial_delay=0, relax_delay=0.02)


@pytest.mark.parametrize("post, spacing", [(0.0005, 48), (0.001, 49)])
@pytest.mark.parametrize("block_mode", [False, True])
@pytest.mark.parametrize("timeline", [None, "drive"])
def test_pulse_ceil_covers_actual_playback_plus_post_once(post, spacing, block_mode, timeline):
    pulse = Pulse("drive", _pulse_cfg(post=post), block_mode=block_mode)

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.emit(pulse, timeline=timeline, tag="first")
        prog.emit(pulse, timeline=timeline, tag="second")

    prog = _program([pulse], body)
    # 39 generator clocks at 384 MHz = 43.68 tProc cycles.
    # pre rounds to 4. Posts are .21504/.43008 cycles: final ceil is 48/49.
    # Separately ceiling end and post gives 49 in both cases (wrong for .0005).
    # Nearest rounding 48.11008 gives 48, omitting part of the .001 post.
    # Rule a instead ceils pre/playback/post separately: 5+44+1=50.
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[4, spacing + 4]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [2 * spacing] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [2 * spacing + 9] * 2
    assert prog.get_time_param("second", "t") * 430.08 == pytest.approx(spacing + 4)
    prog.compile()
    assert shots(prog) == result
