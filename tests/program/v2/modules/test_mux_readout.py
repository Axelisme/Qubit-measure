"""Fixed mux compile, hardware validation and averaged-I/Q selection contracts."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import TypeAdapter
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import ModuleCfg, ModuleCfgFactory, readout
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Repeat

from tests.program.v2._mux_support import (
    AcquisitionSoc,
    RecordingProgram,
    core_mux,
    mux_soccfg,
    native_mux,
)


def test_mux_cfg_namespace_preserves_tones_through_readout_dispatch():
    cfg = readout.MuxReadout.Cfg.model_validate(native_mux().to_dict())
    parsed = TypeAdapter(readout.Readout.Cfg).validate_python(cfg.to_dict())
    module = readout.Readout("mux", parsed)
    assert isinstance(module, readout.MuxReadout)
    assert [tone.ro_ch for tone in module.cfg.tones] == [4, 1]


def test_native_cfg_build_and_union_preserve_order_and_copy():
    core = core_mux()
    cfg = native_mux()
    for parsed in (
        ModuleCfgFactory.from_raw(core.to_dict()),
        TypeAdapter(ModuleCfg).validate_python(core.to_dict()),
        TypeAdapter(readout.ReadoutCfg).validate_python(core.to_dict()),
    ):
        assert isinstance(parsed, readout.MuxReadoutCfg)
        module = readout.Readout("mux", parsed)
        assert isinstance(module, readout.MuxReadout)
        parsed.tones.reverse()
        assert [tone.ro_ch for tone in module.cfg.tones] == [4, 1]
    with pytest.raises(ValueError, match="rebuild"):
        cfg.set_param("freq", 1200.0)


@pytest.mark.parametrize("channels", [(4,), (4, 1, 7, 0, 6, 2, 5, 3)])
def test_real_compile_and_acquire_all_tones(channels):
    cfg = native_mux(channels)
    module = cfg.build("mux")
    soccfg = mux_soccfg()
    prog = RecordingProgram(soccfg, [module], sweep=[("offset", 3)])
    assert prog.gen_declarations == [
        dict(
            ch=0,
            nqz=1,
            mixer_freq=1000.0,
            ro_ch=channels[0],
            mux_freqs=[t.freq for t in cfg.tones],
            mux_gains=[t.gain for t in cfg.tones],
            mux_phases=[t.phase for t in cfg.tones],
        )
    ]
    assert prog.adc_declarations == [
        dict(ch=t.ro_ch, length=cfg.ro_length, freq=t.freq, phase=0.0, gen_ch=0)
        for t in cfg.tones
    ]
    pulse = prog.pulse_definitions[0]
    assert pulse == dict(
        ch=0,
        name="mux",
        style="const",
        mask=list(range(len(channels))),
        length=cfg.length,
    )
    assert prog.triggers == [(list(channels), dict(t=cfg.trig_offset))]
    assert prog.playbacks == [dict(ch=0, name="mux", t=cfg.pre_delay)]
    soc = AcquisitionSoc(soccfg.get_cfg())
    raw = prog.acquire(soc, progress=False)
    assert raw is not None
    iq = module.collect_iq(prog, raw)
    np.testing.assert_allclose(
        iq, np.tile([[ch + 1, -ch - 1] for ch in channels], (3, 1, 1))
    )
    assert len(soc.gen_configs[0][1]) == len(channels)
    assert soc.pfb_configs[0][0] == "pfb0"
    assert soc.pfb_configs[0][2] is None
    assert [r["pfb_port"] for r in soc.pfb_configs[0][1]] == list(channels)


@pytest.mark.parametrize("updates", [dict(length=0.537), dict(ro_length=1.337)])
def test_blocking_covers_rounded_generator_and_adc_windows(updates):
    cfg = native_mux(**updates)
    module = cfg.build("mux")
    prog = RecordingProgram(mux_soccfg(), [module])
    # Scheduled scalar starts round to tProc ticks, hardware windows to their clocks.
    tclk = prog.soccfg["tprocs"][0]["f_time"]

    def ceil_t(t):
        return np.ceil(t * tclk) / tclk

    gen_duration = prog.cycles2us(prog.us2cycles(cfg.length, gen_ch=0), gen_ch=0)
    adc_duration = prog.cycles2us(prog.us2cycles(cfg.ro_length, ro_ch=4), ro_ch=4)
    expected = max(
        ceil_t(ceil_t(cfg.pre_delay) + gen_duration + ceil_t(cfg.post_delay)),
        ceil_t(ceil_t(cfg.trig_offset) + adc_duration),
    )
    assert module.total_length(prog) == pytest.approx(expected)
    assert module.run(prog, 2.0) == pytest.approx(2.0 + expected)
    assert module.allow_rerun()


@pytest.mark.parametrize(
    "fault,field,reason",
    [
        ("generator", "mux.gen_ch", "mixmux8"),
        ("adc", "mux.tones[1].ro_ch", "static PFBv4"),
    ],
)
def test_wiring_rejection_names_the_configuration_field(fault, field, reason):
    soccfg = mux_soccfg()
    if fault == "generator":
        soccfg["gens"][0]["type"] = "axis_signal_gen_v6"
    else:
        soccfg["readouts"][1]["ro_type"] = "axis_pfb_readout_v3"
    with pytest.raises(ValueError, match=reason) as error:
        RecordingProgram(soccfg, [native_mux().build("mux")])
    assert field in str(error.value)
    assert reason in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "axis_signal_gen_v6"),
        ("has_mixer", False),
        ("has_gain", False),
        ("has_phase", False),
        ("n_tones", 1),
    ],
)
def test_invalid_generator_capabilities_fail(field, value):
    cfg = native_mux()
    soccfg = mux_soccfg()
    soccfg["gens"][0][field] = value
    with pytest.raises(ValueError, match="mux"):
        RecordingProgram(soccfg, [cfg.build("mux")])


@pytest.mark.parametrize(
    "field,value",
    [
        ("ro_type", "axis_pfb_readout_v3"),
        ("tproc_ctrl", 0),
        ("ro_fullpath", "other"),
        ("pfb_nout", 1),
        ("pfb_port", 8),
        ("pfb_port", 4),
    ],
)
def test_invalid_pfb_wiring_or_capacity_fails(field, value):
    cfg = native_mux()
    soccfg = mux_soccfg()
    soccfg["readouts"][1][field] = value
    with pytest.raises(ValueError, match="mux"):
        RecordingProgram(soccfg, [cfg.build("mux")])


@pytest.mark.parametrize(
    "fault",
    ["gen_missing", "ro_missing", "duplicate_channel", "short_gen", "short_adc"],
)
def test_invalid_routes_or_rounded_windows_fail(fault):
    cfg = native_mux()
    if fault == "gen_missing":
        cfg.gen_ch = 99
    elif fault == "ro_missing":
        cfg.tones[0].ro_ch = 99
    elif fault == "duplicate_channel":
        cfg.tones[1].ro_ch = 4
    elif fault == "short_gen":
        cfg.length = 1e-8
    elif fault == "short_adc":
        cfg.ro_length = 1e-8
    with pytest.raises(ValueError, match="mux"):
        RecordingProgram(mux_soccfg(), [cfg.build("mux")])


class UnrelatedReadout(Module):
    name = "unrelated"

    def init(self, prog):
        prog.declare_readout(ch=8, length=0.2)
        prog.add_readoutconfig(ch=8, name="unrelated", freq=100.0)

    def run(self, prog, t=0.0):
        prog.send_readoutconfig(8, "unrelated", t=t)
        prog.trigger([8])
        return t + 0.3


@pytest.mark.parametrize("unrelated_first", [True, False])
def test_collect_iq_uses_current_declarations_and_read_index(unrelated_first):
    module = native_mux().build("mux")
    modules = (
        [UnrelatedReadout(), module]
        if unrelated_first
        else [module, UnrelatedReadout()]
    )
    prog = ModularProgramV2(mux_soccfg(), modules)
    selected = {
        4: np.arange(24, dtype=np.float64).reshape(2, 2, 3, 2),
        1: 100 + np.arange(24, dtype=np.float64).reshape(2, 2, 3, 2),
        8: np.zeros((1, 2), dtype=np.float64),
    }
    raw = [selected[ch] for ch in prog.ro_chs]
    selected[4][1, 0, 0, 0] = np.nan
    before = deepcopy(raw)
    iq = module.collect_iq(prog, raw, read_index=1)
    assert iq.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(iq[..., 0, :], selected[4][1])
    np.testing.assert_array_equal(iq[..., 1, :], selected[1][1])
    for actual, expected in zip(raw, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "fault",
    [
        "uninitialized",
        "missing_channel",
        "count",
        "negative",
        "bool",
        "bounds",
        "no_read",
        "iq_width",
        "read_shape",
        "sweep_shape",
    ],
)
def test_collect_iq_rejects_invalid_averaged_input(fault):
    module = native_mux().build("mux")
    prog = RecordingProgram(mux_soccfg(), [module])
    raw: list[NDArray[np.float64]] = [
        np.zeros((2, 3, 2), dtype=np.float64) for _ in range(2)
    ]
    index = 0
    if fault == "uninitialized":
        module = native_mux().build("mux")
    elif fault == "missing_channel":
        prog = RecordingProgram(mux_soccfg(), [])
    elif fault == "count":
        raw.pop()
    elif fault == "negative":
        index = -1
    elif fault == "bool":
        index = True
    elif fault == "bounds":
        index = 2
    elif fault == "no_read":
        raw[0] = np.zeros(2)
    elif fault == "iq_width":
        raw[0] = np.zeros((2, 3, 3))
    elif fault == "read_shape":
        raw[0] = np.zeros((3, 3, 2))
    elif fault == "sweep_shape":
        raw[0] = np.zeros((2, 4, 2))
    with pytest.raises(ValueError, match="mux"):
        module.collect_iq(prog, raw, read_index=index)


def test_collect_iq_rejects_fractional_read_index_at_runtime():
    module = native_mux().build("mux")
    prog = RecordingProgram(mux_soccfg(), [module])
    raw = [np.zeros((2, 3, 2), dtype=np.float64) for _ in range(2)]
    # Deliberately violate static index typing to verify rejection at the public runtime boundary.
    public_method = "collect_iq"
    collect = getattr(module, public_method)
    with pytest.raises(ValueError, match="mux.*read_index"):
        collect(prog, raw, read_index=0.5)


def test_collect_iq_rejects_float32_at_runtime():
    module = native_mux().build("mux")
    prog = RecordingProgram(mux_soccfg(), [module])
    raw = [np.zeros((2, 3, 2), dtype=np.float32) for _ in range(2)]
    # Deliberately violate static dtype typing: QICK's runtime dtype has no static guarantee.
    public_method = "collect_iq"
    collect = getattr(module, public_method)
    with pytest.raises(ValueError, match=r"mux.*tone 0.*ro_ch 4.*acquired\[0\]"):
        collect(prog, raw)


class RawBufferProgram(ModularProgramV2):
    """Supply synthetic acquisition buffers through the public getter boundary."""

    raw: list[NDArray[np.int64]] | None = None

    def get_raw(self) -> list[NDArray[np.int64]] | None:
        return self.raw


def raw_buffer_program(preparations=4):
    module = native_mux().build("mux")
    repeated = Repeat("reads", 2).add_content(module)
    prog = RawBufferProgram(
        mux_soccfg(),
        [UnrelatedReadout(), repeated],
        sweep=[("preparation", preparations), ("point", 2)],
        reps=3,
    )
    # QICK counts a trigger macro once even when Repeat executes it twice.
    prog.set_reads_per_shot(2)
    # Each loop/read has a distinct sentinel. The unrelated channel deliberately
    # has an irrelevant shape; it must not constrain the selected tones.
    sentinels = np.arange(3 * preparations * 2 * 2 * 2, dtype=np.int64).reshape(
        3, preparations, 2, 2, 2
    )
    by_channel = {
        8: np.zeros((1, 2), dtype=np.int64),
        4: sentinels * prog.ro_chs[4]["length"],
        1: (1000 + sentinels) * prog.ro_chs[1]["length"],
    }
    prog.raw = [by_channel[ch] for ch in prog.ro_chs]
    return module, prog, sentinels


@pytest.mark.parametrize("preparations", [2, 4])
def test_collect_raw_iq_preserves_loops_reads_and_tone_shot_pairing(preparations):
    module, prog, sentinels = raw_buffer_program(preparations)
    before = deepcopy(prog.raw)
    iq = module.collect_raw_iq(prog)
    assert iq.dtype == np.float64
    assert iq.shape == (3, preparations, 2, 2, 2, 2)
    np.testing.assert_array_equal(iq[..., 0, :], sentinels)
    np.testing.assert_array_equal(iq[..., 1, :], 1000 + sentinels)
    selected = module.collect_raw_iq(prog, read_index=1)
    np.testing.assert_array_equal(selected, iq[..., 1, :, :])
    np.testing.assert_array_equal(module.collect_raw_iq(prog), iq)
    assert prog.raw is not None and before is not None
    for actual, expected in zip(prog.raw, before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    iq[...] = -100
    np.testing.assert_array_equal(module.collect_raw_iq(prog, read_index=1), selected)


@pytest.mark.parametrize("read_index", [-1, True, 2, 0.5])
def test_collect_raw_iq_rejects_invalid_read_index(read_index):
    module, prog, _ = raw_buffer_program()
    # Include runtime-invalid index types without weakening the public signature.
    public_method = "collect_raw_iq"
    collect = getattr(module, public_method)
    with pytest.raises(ValueError, match="mux.*read_index"):
        collect(prog, read_index=read_index)


@pytest.mark.parametrize("fault", ["missing", "count", "loop", "read", "iq", "dtype"])
def test_collect_raw_iq_rejects_invalid_buffers(fault, monkeypatch):
    module, prog, _ = raw_buffer_program()
    assert prog.raw is not None
    if fault == "missing":
        prog.raw = None
    elif fault == "count":
        prog.raw.pop()
    elif fault == "loop":
        prog.raw[-1] = np.zeros((3, 3, 2, 2, 2), dtype=np.int64)
    elif fault == "read":
        prog.raw[-1] = np.zeros((3, 4, 2, 1, 2), dtype=np.int64)
    elif fault == "iq":
        prog.raw[-1] = np.zeros((3, 4, 2, 2, 3), dtype=np.int64)
    elif fault == "dtype":
        # Only the external getter is replaced, not raw conversion/validation.
        bad_raw = [array.astype(np.float64) for array in prog.raw]
        monkeypatch.setattr(prog, "get_raw", lambda: bad_raw)
    with pytest.raises(ValueError, match="mux"):
        module.collect_raw_iq(prog)


@pytest.mark.parametrize("fault", ["uninitialized", "other_program", "missing_channel"])
def test_collect_raw_iq_requires_initializing_program_and_channels(fault):
    module, prog, _ = raw_buffer_program()
    if fault == "uninitialized":
        module = native_mux().build("mux")
    elif fault == "other_program":
        _, prog, _ = raw_buffer_program()
    elif fault == "missing_channel":
        del prog.ro_chs[4]
    with pytest.raises(ValueError, match="mux"):
        module.collect_raw_iq(prog)


class RouteReuse(Module):
    name = "other"

    def __init__(self, route: str):
        self.route = route

    def init(self, prog):
        if self.route == "gen":
            prog.declare_gen(
                ch=0,
                nqz=1,
                mixer_freq=1000.0,
                mux_freqs=[1000.0, 1080.0],
                mux_gains=[0.1, 0.1],
                mux_phases=[0.0, 45.0],
                ro_ch=4,
            )
        else:
            prog.declare_readout(ch=4, length=0.413, freq=1000.0, gen_ch=0)

    def run(self, prog, t=0.0):
        return t


@pytest.mark.parametrize("route", ["gen", "readout"])
@pytest.mark.parametrize("nested", [False, True])
def test_later_module_cannot_overwrite_mux_routes(route, nested):
    module = native_mux().build("mux")
    first = Repeat("repeat", 2).add_content(module) if nested else module
    channel = 0 if route == "gen" else 4
    with pytest.raises(ValueError, match=f"mux.*{route}.*route.*{channel}"):
        RecordingProgram(mux_soccfg(), [first, RouteReuse(route)])


@pytest.mark.parametrize("collision", ["gen", "readout"])
def test_routes_cannot_be_silently_redeclared(collision):
    first = native_mux((4,)).build("first")
    second_cfg = native_mux((1,) if collision == "gen" else (4,))
    if collision == "readout":
        soccfg = mux_soccfg()
        soccfg["gens"][1].update(soccfg["gens"][0])
        soccfg["gens"][1]["tproc_ch"] = 1
        second_cfg.gen_ch = 1
    else:
        soccfg = mux_soccfg()
    with pytest.raises(ValueError, match="second"):
        RecordingProgram(soccfg, [first, second_cfg.build("second")])
