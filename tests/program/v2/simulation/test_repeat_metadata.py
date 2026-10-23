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
        ("swept", "ce07cccf6884cd443d72ac94ba6cb734d43c9950b9b9cae00e8657e646bb3860"),
        (
            "nonblocking",
            "a6ab6c7947682738559e29c869d593da18de47a617fbe73cb377406daf7749b5",
        ),
        (
            "flat_top",
            "2e4ade443db4157b0481f9ebb891eef269686799a1ee1c1c448578bcd543a3a0",
        ),
        (
            "computed",
            "03bb5211ed5da8dce0d3f86d70626e3538f48e62a1bdc401e34e0b2a2fd41d8a",
        ),
        (
            "computed_selection",
            "d796deeb3c2001c266b1de9e5dcb4f3051557424745ba70594d532d7328d181f",
        ),
        (
            "computed_tables",
            "c479a8be410544303e9653960329862f1a59fcebca7caabb0155be5c49183445",
        ),
        (
            "computed_qick_time",
            "dd7d3f7318c487daf917707acb69f0d9bda0e9b49d5c91b01feebc9872646c11",
        ),
        (
            "computed_flat_top",
            "7d08fa57c36b1fd498be2150abf7ebc87bb8568e91fa7fa46923ccfab10bf182",
        ),
        ("dynamic", "58fbed217601aeaa18033bccd5bec43aeec50f268ed4806604be0dae2cc47291"),
        ("branch", "283718448f03e2272ffe4ed594bc92af6a5028bcf9e19809a1a0fae6eb5fb9a7"),
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
