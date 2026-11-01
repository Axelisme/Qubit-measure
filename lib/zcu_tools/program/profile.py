"""Build compile-only QICK configurations from independent board settings."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from qick import QickConfig

_PositiveInt = Annotated[int, Field(gt=0)]
_Count = Annotated[int, Field(ge=0)]
_PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Tproc(_Settings):
    type: str
    revision: _PositiveInt
    f_time: _PositiveFloat
    f_core: _PositiveFloat
    pmem_size: _PositiveInt
    wmem_size: _PositiveInt
    dmem_size: _PositiveInt
    dreg_qty: _PositiveInt
    output_pins: list[object]
    start_pin: str
    stop_pin: str


class _ClockGroup(_Settings):
    type: str
    fs_mult: _PositiveInt
    fs_div: _PositiveInt
    samps_per_clk: _PositiveInt
    b_dds: _PositiveInt

    def sample_rate(self, refclk: float) -> float:
        """Return this group's RF sample rate in MHz for the given reference."""
        return refclk * self.fs_mult / self.fs_div


class _Generator(_ClockGroup):
    dac: Annotated[str, Field(pattern=r"^[0-9]{2}$")]
    tproc_ch: _Count
    tmux_ch: _Count | None
    interpolation: _PositiveInt
    fabric_div: _PositiveInt
    rf_fabric_div: _PositiveInt
    b_phase: _PositiveInt
    maxlen: _PositiveInt
    complex_env: bool
    has_dds: bool
    has_mixer: bool
    maxv: _PositiveInt
    maxv_scale: _PositiveFloat


class _Readout(_ClockGroup):
    adc: Annotated[str, Field(pattern=r"^[0-9]{2}$")]
    tproc_ctrl: _Count | None
    tmux_ch: _Count | None
    tproc_ch: Annotated[int, Field(ge=-1)]
    b_phase: _PositiveInt | None
    trigger_bit: _Count
    ro_type: str
    decimation: _PositiveInt
    buf_maxlen: _PositiveInt
    avg_maxlen: _PositiveInt
    avgbuf_type: str
    avgbuf_version: str
    has_edge_counter: bool
    has_weights: bool
    has_outsel: bool
    iq_offset: Annotated[float, Field(allow_inf_nan=False)]
    trigger_type: Literal["dport", "tport"]
    trigger_port: _Count
    coupling: str


class _RF(_Settings):
    dac_power: str
    clk_groups: list[object]


class _MuxGenerator(_Settings):
    type: str
    has_mixer: bool
    n_tones: _PositiveInt
    has_gain: bool
    has_phase: bool
    interpolation: _PositiveInt


class _PFB(_Settings):
    type: str
    ro_type: str
    ro_fullpath: str
    pfb_nout: _PositiveInt
    pfb_nch: Annotated[int, Field(gt=0, multiple_of=2)]
    pfb_ch_offset: _Count
    pfb_dds_on_output: bool
    has_outsel: bool


class _Mux(_Settings):
    gen_ch: _Count
    ro_chs: Annotated[list[_Count], Field(min_length=1)]
    generator: _MuxGenerator
    readout: _PFB


class _Profile(_Settings):
    board: str
    sw_version: str
    fw_timestamp: str
    refclk_freq: _PositiveFloat
    extra_description: list[str]
    tprocs: Annotated[list[_Tproc], Field(min_length=1)]
    generators: list[_Generator]
    readouts: list[_Readout]
    rf: _RF
    mux: list[_Mux]


def _validate_wave_ports(profile: _Profile) -> None:
    users = [
        (f"generators.{index}.tproc_ch", channel.tproc_ch, channel.tmux_ch)
        for index, channel in enumerate(profile.generators)
    ]
    users.extend(
        (f"readouts.{index}.tproc_ctrl", channel.tproc_ctrl, channel.tmux_ch)
        for index, channel in enumerate(profile.readouts)
        if channel.tproc_ctrl is not None
    )
    ports: dict[int, set[int | None]] = {}
    for path, port, selector in users:
        selectors = ports.setdefault(port, set())
        if selectors and (selector is None or None in selectors or selector in selectors):
            raise ValueError(f"{path}: shared wave-output port requires distinct tmux_ch")
        selectors.add(selector)


def _generators(
    profile: _Profile,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    gens: list[dict[str, object]] = []
    dacs: dict[str, dict[str, object]] = {}
    for index, channel in enumerate(profile.generators):
        if channel.dac in dacs:
            raise ValueError(f"generators.{index}.dac: requires a unique DAC name")
        fs = channel.sample_rate(profile.refclk_freq)
        cfg = channel.model_dump(
            exclude={"fabric_div", "rf_fabric_div"}, exclude_none=True
        )
        cfg.update(
            fs=fs,
            f_dds=fs / channel.interpolation,
            f_fabric=fs / channel.fabric_div,
            fdds_div=channel.fs_div * channel.interpolation,
        )
        gens.append(cfg)
        dacs[channel.dac] = {
            "fs": fs,
            "fs_mult": channel.fs_mult,
            "fs_div": channel.fs_div,
            "interpolation": channel.interpolation,
            "f_fabric": fs / channel.rf_fabric_div,
        }
    return gens, dacs


def _readouts(
    profile: _Profile,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    readouts: list[dict[str, object]] = []
    adcs: dict[str, dict[str, object]] = {}
    for index, channel in enumerate(profile.readouts):
        fs = channel.sample_rate(profile.refclk_freq)
        f_output = fs / (channel.decimation * channel.samps_per_clk)
        cfg = channel.model_dump(
            exclude={"samps_per_clk", "coupling"}, exclude_none=True
        )
        cfg.update(
            fs=fs,
            f_dds=fs / channel.decimation,
            f_output=f_output,
            fdds_div=channel.fs_div * channel.decimation,
        )
        readouts.append(cfg)
        adc: dict[str, object] = {
            "fs": fs,
            "fs_mult": channel.fs_mult,
            "fs_div": channel.fs_div,
            "decimation": channel.decimation,
            "f_fabric": f_output,
            "coupling": channel.coupling,
        }
        if channel.adc in adcs and adcs[channel.adc] != adc:
            raise ValueError(
                f"readouts.{index}.adc: shared ADC requires identical RF settings"
            )
        adcs[channel.adc] = adc
    return readouts, adcs


def _apply_mux(
    profile: _Profile,
    gens: list[dict[str, object]],
    readouts: list[dict[str, object]],
) -> None:
    used_gens: set[int] = set()
    used_readouts: set[int] = set()
    for index, mux in enumerate(profile.mux):
        path = f"mux.{index}"
        if mux.gen_ch >= len(gens) or mux.gen_ch in used_gens:
            raise ValueError(f"{path}.gen_ch: requires an unused in-range generator")
        if (
            len(mux.ro_chs) > mux.readout.pfb_nout
            or len(set(mux.ro_chs)) != len(mux.ro_chs)
            or any(ch >= len(readouts) or ch in used_readouts for ch in mux.ro_chs)
        ):
            raise ValueError(
                f"{path}.ro_chs: requires unique unused in-range ADCs within pfb_nout"
            )
        if any(profile.readouts[ch].tproc_ctrl is not None for ch in mux.ro_chs):
            raise ValueError(f"{path}.ro_chs: mux readouts require tproc_ctrl: null")
        used_gens.add(mux.gen_ch)
        used_readouts.update(mux.ro_chs)
        gen = gens[mux.gen_ch]
        gen_group = profile.generators[mux.gen_ch]
        gen.update(mux.generator.model_dump())
        gen.update(
            f_dds=gen_group.sample_rate(profile.refclk_freq)
            / mux.generator.interpolation,
            fdds_div=gen_group.fs_div * mux.generator.interpolation,
        )
        for port, ch in enumerate(mux.ro_chs):
            ro = readouts[ch]
            ro.update(mux.readout.model_dump())
            ro_group = profile.readouts[ch]
            fs = ro_group.sample_rate(profile.refclk_freq)
            output_div = ro_group.decimation * (mux.readout.pfb_nch // 2)
            dds_div = (
                output_div if mux.readout.pfb_dds_on_output else ro_group.decimation
            )
            ro.update(
                pfb_port=port,
                f_output=fs / output_div,
                f_dds=fs / dds_div,
                fdds_div=ro_group.fs_div * dds_div,
            )


def build_soccfg(profile: Mapping[str, object]) -> QickConfig:
    """Build a detached QickConfig from caller-loaded board settings.

    The mapping supplies board/software metadata, refclk_freq in MHz, tprocs,
    ordered generators/readouts, rf metadata and a mux list (empty if unused).
    Each generator supplies dac (two decimal digits), tproc_ch (wave-output
    port) and tmux_ch (nonnegative selector or explicit None). Each readout
    supplies adc (two decimal digits), tproc_ctrl (wave-output port or None),
    tmux_ch (selector or None), trigger_type ('dport' or 'tport'), trigger_port,
    trigger_bit, tproc_ch (readback input, -1 if absent) and b_phase (positive
    bits or None). Ports/selectors/bits are nonnegative integers. None keys
    are omitted from QICK channel entries; all input fields remain required.

    Channels supply capability fields and positive integer clock ratios.
    Sample rate is refclk_freq * fs_mult / fs_div. Generator DDS divides by
    interpolation, logical fabric by fabric_div and RF fabric by rf_fabric_div.
    Readout DDS divides by decimation and output by decimation*samps_per_clk.
    Mux blocks select gen_ch and ordered ro_chs, override capabilities and
    derive PFB output/DDS clocks from pfb_nch and pfb_dds_on_output. Mux readouts
    must have tproc_ctrl=None; trigger wiring and per-readout b_phase stay intact.

    Duplicate DACs, inconsistent shared ADC clocks/metadata and ambiguous shared
    wave-output ports raise ValueError at the later user's dotted field path.
    Shared ports require distinct non-None tmux_ch values across generators and
    controlled readouts. Unknown, derived, missing or invalid fields also raise
    ValueError with dotted paths. No files, hardware, installed-version metadata
    or environment state are read. Input and output share no mutable values.
    """
    settings = _Profile.model_validate(profile)
    gens, dacs = _generators(settings)
    readouts, adcs = _readouts(settings)
    _apply_mux(settings, gens, readouts)
    _validate_wave_ports(settings)
    cfg = settings.model_dump(exclude={"generators", "readouts", "rf", "mux"})
    cfg.update(
        gens=gens,
        readouts=readouts,
        rf={**settings.rf.model_dump(), "dacs": dacs, "adcs": adcs},
    )
    return QickConfig(cfg)
