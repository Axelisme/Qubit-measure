"""Public acquisition rejects reps live-ins and missing routes."""

import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._profiles import load_soccfg
from tests.program._timeline import compile_modules


def drive(*, ch=1, waveform=None):
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=80,
        gain=-0.3,
        waveform=waveform or ConstWaveformCfg(length=0.1),
    )


def receiver():
    return PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )


@pytest.mark.parametrize("dependency", ["sweep", "register", "branch"])
def test_reps_dependent_timeline_rejected_before_backend(dependency):
    env = white_noise(
        load_soccfg(n_gens=2), std_adc=0, routes=((0, 0), (1, 0)), seed=None
    )
    cfg = drive()
    builder = TimelineBuilder(env.soccfg, reps=2)
    if dependency == "sweep":
        cfg.gain = QickParam(0.1, {"reps": 0.2})
        builder.emit(cfg.build("drive"))
    elif dependency == "register":
        builder.emit(LoadValue("count", [1, 2], "reps", "n"))
        with builder.repeat("inner", "n", range_hint=(1, 2)):
            builder.emit(cfg.build("drive"))
    else:
        with builder.branch(by="reps", align_timeline=True) as cases:
            for index in range(2):
                with cases.case(index):
                    builder.emit(cfg.build(f"drive_{index}"))
    builder.emit(receiver().build("readout"))
    builder.delay_by_timeline()
    with pytest.raises(ValueError, match=r"(?:reps|cross-shot|measurement)"):
        builder.build().acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


@pytest.mark.parametrize("missing", ["generator", "route"])
def test_soc_rejects_unbound_generator_or_adc_route_before_backend(missing):
    env = white_noise(
        load_soccfg(n_gens=2, n_readouts=2),
        std_adc=0,
        routes=((0, 0),) if missing == "generator" else ((0, 0), (1, 0)),
        seed=None,
    )
    if missing == "generator":
        modules = [drive().build("unwired_drive"), receiver().build("readout")]
    else:
        cfg = receiver()
        cfg.gen_ch = 1
        cfg.ro_ch = 1
        modules = [cfg.build("readout")]
    prog = compile_modules(env.soccfg, modules=modules)
    with pytest.raises(
        ValueError,
        match="generator 1 is not bound"
        if missing == "generator"
        else "ADC 1.*no wired generator route",
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds
