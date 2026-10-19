"""Bitwise frontend characterization for repeated pulse metadata.

Digests cover the full numeric request, including inactive placeholders, candidate
arrays and lowered flat-top envelopes. Semantic checks make the Repeat contract
visible without depending on the interpreter's cache or private helpers.
"""

from dataclasses import fields, is_dataclass
from hashlib import sha256

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.control import Branch, Repeat
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg, TableLengthPulse
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.simulation.frontend import build_request


@pytest.fixture
def environment():
    env = white_noise(std_adc=0, seed=4, n_gens=2, routes=((0, 0), (1, 0)))
    yield env
    env.devices.close_all_devices()


def repeated_program(env, case):
    """Exercise changing masks/timestamps with one initialized pulse per Repeat."""
    cfg = PulseCfg(
        ch=1,
        nqz=1,
        freq=QickParam(80, {"axis": 13.1}),
        phase=QickParam(3, {"axis": 37}),
        gain=QickParam(-0.3, {"axis": 0.1}),
        pre_delay=QickParam(0.013, {"axis": 0.031}),
        post_delay=0.017,
        waveform=ConstWaveformCfg(length=QickParam(0.1, {"axis": 0.04})),
    )
    if case == "flat_top":
        cfg.waveform = FlatTopWaveformCfg(
            length=QickParam(0.1078125, {"axis": 0.015625}),
            raise_waveform=GaussWaveformCfg(length=0.1, sigma=0.02),
        )
    if case.startswith("computed") or case in ("dynamic", "branch", "nonblocking"):
        cfg.pre_delay = 0.013
        cfg.waveform = ConstWaveformCfg(length=0.1)
    pulse: Module = Pulse("drive", cfg, block_mode=case != "nonblocking")
    if case.startswith("computed"):
        shaped = cfg.with_updates(waveform=GaussWaveformCfg(length=0.1, sigma=0.02))
        candidates = [cfg, shaped, cfg]
        if case != "computed":
            candidates = [
                cfg,
                cfg.with_updates(waveform=GaussWaveformCfg(length=0.27, sigma=0.04)),
                cfg.with_updates(waveform=ConstWaveformCfg(length=0.055)),
            ]
        if case == "computed_flat_top":
            candidates = [
                cfg.with_updates(
                    waveform=FlatTopWaveformCfg(
                        length=length,
                        raise_waveform=GaussWaveformCfg(length=0.05, sigma=0.01),
                    )
                )
                for length in (0.06, 0.15, 0.23)
            ]
        pulse = ComputedPulse(
            "gate",
            val_reg="axis" if case == "computed" else "selected",
            pulses=candidates,
        )
    elif case == "dynamic":
        cfg.waveform = FlatTopWaveformCfg(
            length=0.1,
            raise_waveform=GaussWaveformCfg(length=0.05, sigma=0.01),
        )
        pulse = TableLengthPulse(
            "dynamic", cfg, lengths=[0.06, 0.12, 0.2], idx_reg="axis"
        )
    elif case == "branch":
        pulse = Branch(
            "choice", pulse, cfg.build("other"), cfg.build("third"), compare_by="axis"
        )
    body: list[Module] = [
        SoftDelay(
            "before",
            QickParam(0.017, {"axis": 0.019})
            if case == "computed_qick_time"
            else 0.017,
        ),
        pulse,
        SoftDelay("after", 0.009),
    ]
    if case.startswith("computed_"):
        body.insert(0, LoadValue("selection", [2, 0], "inner", "selected"))
    if case == "computed_tables":
        body.append(ComputedPulse("second_gate", val_reg="inner", pulses=[cfg, cfg]))
    repeated = Repeat("outer", "count", range_hint=(0, 2)).add_content(
        Repeat("inner", 2).add_content(body)
    )
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    return ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=2),
        modules=[
            LoadValue("counts", [0, 2, 1], "axis", "count"),
            repeated,
            readout.build("readout"),
        ],
        sweep=[("axis", 3)],
    )


def request_digest(request):
    """Hash values and exact ndarray bytes, never object identity or aliasing."""
    digest = sha256()

    def visit(value):
        if isinstance(value, np.ndarray):
            digest.update(repr((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                digest.update(field.name.encode())
                visit(getattr(value, field.name))
        elif isinstance(value, tuple):
            digest.update(str(len(value)).encode())
            for child in value:
                visit(child)
        else:
            digest.update(repr(value).encode())

    visit(request)
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("swept", "0aed6216d4df3f0bef2193d314a2c111f9583da1a1f737fc63eb214d5060e876"),
        (
            "nonblocking",
            "b2d818230ccbcb04640895080495e3b0587c92964c33dc3bf7111755dcb098b8",
        ),
        (
            "flat_top",
            "eb6d2096581689c4beba05b757e43ca46c99410d098f128f0c4e32c59f5fb4c3",
        ),
        (
            "computed",
            "61e2c12493fa3cc761475eac4587e9817212103ba88f71c234289d990d838cfa",
        ),
        (
            "computed_selection",
            "3240022eff6fb4b38a35122acc0bacec1a5e5aca8359293e94c71ab002a38622",
        ),
        (
            "computed_tables",
            "4299fe9767ee06e0dee1ac10c5f08819983b791de04f02718b532b910c670258",
        ),
        (
            "computed_qick_time",
            "6d2b3d2f2cd9daa7d707a44ac13b1fdcf0d09fe9f0575e48c4e6035537a2962a",
        ),
        (
            "computed_flat_top",
            "f0131f43bad6c34ba461d0ec82b795c5a1f6a364b8829dbc060820d7ec5be3e2",
        ),
        ("dynamic", "5c8ff7cf8fc0c241104b71511ccc609f209b192d980a95cb0491ee66d1c80f89"),
        ("branch", "43cbdc689aed6890279f10dd2ff84ab66ccfea41d5abd40dec7ab55ef0a2c0a9"),
    ],
)
def test_repeated_request_preserves_all_numeric_content(environment, case, expected):
    prog = repeated_program(environment, case)
    request = build_request(prog, "integrated", environment.soc.binding, 0)
    assert request.point_names == ("axis",)
    assert request.point_dims == (3,)
    assert request.reps == 2
    for template in request.templates:
        for slot in template.events:
            if slot.source == "readout":
                continue
            outer, inner = slot.iterations
            assert outer[0] == "outer" and inner[0] == "inner"
            np.testing.assert_array_equal(
                slot.active, np.asarray([0, 2, 1])[template.points] > outer[1]
            )
    if case.startswith("computed_"):
        assert not any(
            event.source == "gate" for event in request.materialize_point(0).events
        )
        for point in (1, 2):
            gates = [
                event
                for event in request.materialize_point(point).events
                if event.source == "gate"
            ]
            for event in gates:
                # The register changes on every inner iteration, not every point.
                inner = dict(event.iterations)["inner"]
                gain = -0.25 if point == 1 else -0.2
                if case == "computed_flat_top":
                    assert any(
                        event.gain == pytest.approx(value, abs=1e-4)
                        for value in (gain, gain / 2)
                    )
                else:
                    assert event.gain == pytest.approx(gain, abs=1e-4)
                if case != "computed_flat_top":
                    assert event.duration_us == pytest.approx(
                        0.055 if inner == 0 else 0.1, abs=0.003
                    )
            if case == "computed_tables":
                second = [
                    event
                    for event in request.materialize_point(point).events
                    if event.source == "second_gate"
                ]
                assert len(second) == len(gates)
                assert second[0].start_us > gates[0].start_us + gates[0].duration_us
    assert request_digest(request) == expected


@pytest.mark.parametrize("case", ["computed", "computed_selection"])
def test_next_request_reads_current_compiled_pulse_content(environment, case):
    prog = repeated_program(environment, case)
    first = build_request(prog, "integrated", environment.soc.binding, 0)
    original = request_digest(first)
    table = prog.envelopes[1]["envs"]
    assert isinstance(table, dict)
    next(iter(table.values()))["data"][:, 1] = 7
    second = build_request(prog, "integrated", environment.soc.binding, 0)
    assert request_digest(second) != original
    assert request_digest(first) == original
    assert request_digest(
        build_request(prog, "integrated", environment.soc.binding, 0)
    ) == request_digest(second)
