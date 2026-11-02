"""Public native compile/request fixture builders for compiled frontend tests."""

import numpy as np
from qick import QickConfig
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.simulation.compiled import build_compiled_request
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import load_soccfg


def program(
    body,
    initialize=None,
    *,
    reps=2,
    routes=((0, 0),),
    n_readouts=1,
    soccfg: QickConfig | None = None,
    trigger_time: float | None = 0,
    config_time: float | None = 0,
    readout_length: float = 0.2,
    initial_delay: float = 0,
    relax_delay: float = 0.1,
    cleanup=None,
    stream_transform=None,
):
    """Compile native callbacks on the supplied board or the ordinary test YAML.

    An explicit soccfg must provide gen0/1 and every ADC declared by callbacks.
    n_readouts selects the ordinary YAML's count only when soccfg is omitted.
    """
    env = white_noise(
        load_soccfg(n_gens=2, n_readouts=n_readouts) if soccfg is None else soccfg,
        std_adc=0,
        routes=routes,
        seed=None,
    )

    class Program(TimelineProgramV2):
        @property
        def compiled_stream(self):
            snapshot = super().compiled_stream
            return snapshot if stream_transform is None else stream_transform(snapshot)

        def _initialize(self, cfg):
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0, "p", style="const", freq=100, phase=30, gain=-0.25, length=0.1
            )
            self.declare_readout(0, length=readout_length)
            self.add_readoutconfig(0, "ro", freq=100, gen_ch=0)
            if initialize is not None:
                initialize(self)

        def _body(self, cfg):
            if config_time is not None:
                self.send_readoutconfig(0, "ro", t=config_time)
            body(self)
            if trigger_time is not None:
                self.trigger([0], t=trigger_time)

        def _cleanup(self, cfg):
            if cleanup is not None:
                cleanup(self)

    return Program(
        env.soccfg, reps=reps, initial_delay=initial_delay, relax_delay=relax_delay
    ), env


def clock_invariance_program(
    soccfg: QickConfig, *, extra_generator: bool
) -> TimelineProgramV2:
    """Compile the fixed gen0/ADC0 clock-invariance input on the supplied board.

    The board must provide generators 0/1 and ADC0. extra_generator declares
    generator 1 without emitting events; callers set its heterogeneous clock.
    Both variants retain the same envelope, pulse, trigger and repetition inputs.
    Compilation errors propagate to the caller.
    """

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare_gen(0, nqz=1)
            if extra_generator:
                self.declare_gen(1, nqz=1)
            self.add_envelope(0, "shape", idata=np.full(48, -8191.0))
            self.add_pulse(
                0,
                "p",
                style="arb",
                envelope="shape",
                freq=100,
                phase=30,
                gain=-0.25,
            )
            self.declare_readout(0, length=0.2)
            self.add_readoutconfig(0, "ro", freq=100, gen_ch=0)

        def _body(self, cfg):
            self.send_readoutconfig(0, "ro", t=0)
            self.pulse(0, "p", t=0.1)
            self.trigger([0], t=0.1)

    return Program(soccfg, reps=2, initial_delay=0, relax_delay=0.1)


def scheduled_shots(prog):
    """Return per-shot event kinds, relative cycles and unsigned wave words."""
    analysis = ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list,
            prog.labels,
            prog.timing_layout,
            dmem=prog.compile_datamem(),
            wmem=ir.compiled_wmem(prog),
        )
    )
    return [
        tuple((e.kind, e.cycle - shot.start_cycle, e.wave_words) for e in shot.events)
        for shot in analysis.shots
    ]


def request(prog, env):
    return build_compiled_request(
        prog, "integrated", env.soc.binding, 7, routes=env.soc.routes
    )
