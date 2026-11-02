"""Timed native entry points select lexical scope through both author frontends."""

from contextlib import nullcontext
from math import ceil
from typing import TypedDict

import pytest
from zcu_tools.program import v2
from zcu_tools.program.v2._timeline_selected import PlaybackChoice, SelectedPlayback

from tests.program._profiles import GEN_MHZ, TPROC_MHZ, load_soccfg
from tests.program._timing_expectations import (
    generator_duration,
    runtime_cycles,
    tproc_ceil,
)
from tests.program.v2.timing_support import shots, waves


class _Placement(TypedDict, total=False):
    timeline: str | None


class _TimedOperation(v2.Module):
    """Synthetic module lets builder replay the same public native calls."""

    name = "timed_operation"

    def __init__(self, operation: str, placement: _Placement):
        self.operation = operation
        self.placement = placement
        self.loader = v2.LoadValue("words", [0], idx_reg="index", val_reg="word")

    def init(self, prog):
        raise NotImplementedError("native-only fixture")

    def run(self, prog, t=0.0):
        raise NotImplementedError("native-only fixture")

    def declare(self, prog: v2.TimelineProgramV2) -> None:
        prog.declare_gen(0, nqz=1)
        prog.add_pulse(
            0, "p", style="const", freq=100, phase=0, gain=0.5, length=39 / GEN_MHZ
        )
        prog.declare_readout(0, length=0.2)
        prog.add_readoutconfig(0, "ro", freq=100)
        prog.add_reg("index")
        prog.add_reg("runtime_word")
        prog.declare(self.loader)

    def lower(self, prog, *, timeline, tag):
        self.play(prog)

    def play(self, prog: v2.TimelineProgramV2) -> None:
        """Leave timeline selection to the timed API, not module.emit."""
        prog.write_reg("index", 0)
        prog.write_reg("runtime_word", 1000)
        t = 10 / TPROC_MHZ
        if self.operation == "send_readoutconfig":
            prog.send_readoutconfig(0, "ro", t=t, **self.placement)
        elif self.operation == "send_readoutconfig_from_regs":
            prog.send_readoutconfig_from_regs(
                0, "ro", freq_reg="runtime_word", t=t, **self.placement
            )
        elif self.operation == "pulse_from_regs":
            prog.pulse_from_regs(0, "p", gain_reg="runtime_word", t=t, **self.placement)
        else:
            relation = self.loader.lower_selected(prog)
            selection = SelectedPlayback(
                relation.selector,
                "word",
                (PlaybackChoice("p"),),
                relation,
            )
            prog.pulse_by_reg(0, "word", t=t, selection=selection, **self.placement)


def _compile_operation(frontend: str, operation: str, context: str):
    placement: _Placement = {}
    if context == "none":
        placement["timeline"] = None
    elif context == "explicit":
        placement["timeline"] = "default"
    module = _TimedOperation(operation, placement)

    def body(prog: v2.TimelineBuilder | v2.TimelineProgramV2) -> None:
        prog.fork("default", into=["outer", "inner"])
        prog.timeline_inc(100 / TPROC_MHZ, timeline="outer")
        prog.timeline_inc(200 / TPROC_MHZ, timeline="inner")
        prog.timeline_inc(300 / TPROC_MHZ, timeline="default")
        scope = nullcontext() if context == "outside" else prog.timeline("outer")
        with scope:
            if context == "restored":
                with prog.timeline("inner"):
                    pass
            nested = prog.timeline("inner") if context == "nested" else nullcontext()
            with nested:
                if isinstance(prog, v2.TimelineBuilder):
                    prog.emit(module)
                else:
                    module.play(prog)
        selected = {"explicit": "default", "nested": "inner"}.get(context, "outer")
        prog.delay_by_timeline(timeline=selected)
        prog.pulse(0, "p", t=0)

    soccfg = load_soccfg(1, 1)
    if frontend == "builder":
        builder = v2.TimelineBuilder(soccfg, initial_delay=0, relax_delay=0)
        body(builder)
        return builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(module)

        def _body(self, cfg):
            body(self)

    return Native(soccfg, initial_delay=0, relax_delay=0)


@pytest.mark.parametrize("frontend", ["builder", "native"])
@pytest.mark.parametrize(
    "operation",
    [
        "send_readoutconfig",
        "send_readoutconfig_from_regs",
        "pulse_from_regs",
        "pulse_by_reg",
    ],
)
@pytest.mark.parametrize(
    "context", ["omitted", "none", "explicit", "outside", "nested", "restored"]
)
def test_timed_entry_points_select_scope_and_preserve_native_placement(
    frontend, operation, context
):
    prog = _compile_operation(frontend, operation, context)
    expected_tick = {"explicit": 310, "outside": 10, "nested": 210}.get(context, 110)
    expected_end = 100 if context == "outside" else expected_tick
    if context != "outside" and operation.startswith("pulse"):
        expected_end += (
            ceil(runtime_cycles(prog.soccfg, 39))
            if operation == "pulse_by_reg"
            else tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, 39 / GEN_MHZ))
        )
    # The second, native pulse observes the selected end after delay_by_timeline. This also
    # covers register-selected ends that have no affine get_time_param value.
    assert [waves(shot) for shot in shots(prog)] == [[expected_tick, expected_end]]
