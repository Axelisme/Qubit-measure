"""Emission order and fail-fast capability checks at public acquisition."""

import numpy as np
import pytest
from qick.asm_v2 import AsmInst, QickParam
from zcu_tools.core.simulation import ChannelSignals, RoundSignals
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Repeat
from zcu_tools.program.v2.modules.delay import Join, SoftDelay
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


class Feedback(Module):
    """Synthetic unsupported operation; compilation itself needs no hardware."""

    def __init__(self):
        self.name = "hidden_feedback"

    def init(self, prog):
        pass

    def run(self, prog, t=0.0):
        return t


class RawProgram(ModularProgramV2):
    """A legal hardware program whose raw ISA is outside semantic simulation."""

    def _body(self, cfg):
        super()._body(cfg)
        self.asm_inst({"CMD": "NOP"})


@pytest.mark.parametrize("method", ["acquire", "acquire_decimated"])
def test_raw_program_body_rejected_instead_of_silently_ignoring_instructions(method):
    env = white_noise(std_adc=0)
    prog = RawProgram(env.soccfg, ProgramV2Cfg(), modules=[readout("readout")])
    with pytest.raises(ValueError, match="RawProgram.*raw.*unsupported"):
        getattr(prog, method)(env.soc, progress=False)
    assert not env.soc.binding.rounds


def test_raw_loop_hook_rejected_before_any_backend_round():
    env = white_noise(std_adc=0)
    prog = ModularProgramV2(
        env.soccfg, ProgramV2Cfg(), modules=[readout("readout")], sweep=[("axis", 2)]
    )
    prog.append_loop_after("axis", AsmInst(inst={"CMD": "NOP"}, addr_inc=1))
    prog.compile()
    with pytest.raises(ValueError, match="axis.*raw.*unsupported"):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


def test_unwired_generator_pulse_is_not_silently_accepted():
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0),))
    drive = PulseCfg(
        ch=1, nqz=1, freq=50, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
    ).build("unwired_drive")
    prog = ModularProgramV2(
        env.soccfg, ProgramV2Cfg(), modules=[drive, readout("readout")]
    )
    with pytest.raises(ValueError, match="unwired_drive.*1.*bound"):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


def readout(name):
    return PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    ).build(name)


@pytest.mark.parametrize("method", ["acquire", "acquire_decimated"])
def test_unknown_unexecuted_module_rejected_before_any_signal_generation(method):
    env = white_noise(std_adc=0)
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(),
        modules=[
            Repeat("never", 0).add_content(Feedback()),
            readout("readout"),
        ],
    )
    with pytest.raises(ValueError, match="Feedback.*hidden_feedback.*unsupported"):
        getattr(prog, method)(env.soc, progress=False)
    assert not env.soc.binding.rounds


def test_swept_trigger_uses_start_and_truncated_loop_step_not_pointwise_rounding():
    env = white_noise(std_adc=0)
    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
        trig_offset=QickParam(0, {"axis": 0.007}),
    )
    prog = ModularProgramV2(
        env.soccfg, ProgramV2Cfg(), modules=[cfg.build("readout")], sweep=[("axis", 3)]
    )
    prog.acquire(env.soc, progress=False)
    request = env.soc.binding.last_request
    assert [t.readouts[0].trigger_us[0] for t in request.timelines] == pytest.approx(
        [0, 1 / 430.08, 2 / 430.08]
    )


def test_join_emits_interleaved_read_order_and_companion_raw_from_same_realization():
    env = white_noise(std_adc=0)
    modules = [
        Join(
            [readout("first"), SoftDelay("later", 0.3), readout("third")],
            [SoftDelay("middle", 0.2), readout("second")],
        )
    ]
    prog = ModularProgramV2(env.soccfg, ProgramV2Cfg(reps=2, rounds=2), modules=modules)
    seen = []

    def execute(request):
        seen.append(request)
        samples = request.acquire.readouts[0].samples
        traces = np.empty((2, 3, samples, 2))
        for execution in range(2):
            for read in range(3):
                traces[execution, read] = [
                    10 * request.index + 3 * execution + read,
                    -read,
                ]
        return RoundSignals((ChannelSignals(0, traces),))

    env.soc.binding.execute = execute
    traces = prog.acquire_decimated(env.soc, progress=False)[0]
    request = seen[0].acquire
    assert [e.source for e in request.controls[0]] == ["first", "second", "third"]
    assert np.all(np.diff(request.timelines[0].readouts[0].trigger_us) > 0)
    assert len(seen) == 2
    assert traces.shape == (2, 3, request.readouts[0].samples, 2)
    np.testing.assert_array_equal(traces[:, :, 0, 0], [[5, 6, 7], [8, 9, 10]])
    raw = prog.get_raw()[0]
    np.testing.assert_array_equal(
        raw[..., 0],
        np.array([[10, 11, 12], [13, 14, 15]]) * request.readouts[0].samples,
    )
