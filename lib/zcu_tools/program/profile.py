"""Build compile-only QICK configurations from independent board settings."""

from collections.abc import Mapping
from typing import Annotated

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
    count: _Count
    type: str
    fs_mult: _PositiveInt
    fs_div: _PositiveInt
    samps_per_clk: _PositiveInt
    b_dds: _PositiveInt

    def sample_rate(self, refclk: float) -> float:
        """Return this group's RF sample rate in MHz for the given reference."""
        return refclk * self.fs_mult / self.fs_div


class _Generator(_ClockGroup):
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
    trigger_type: str
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
    b_phase: _PositiveInt


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


def _generators(
    profile: _Profile,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    gens: list[dict[str, object]] = []
    dacs: dict[str, dict[str, object]] = {}
    for group in profile.generators:
        fs = group.sample_rate(profile.refclk_freq)
        for _ in range(group.count):
            ch = len(gens)
            dac = f"{ch:02d}"
            cfg = group.model_dump(exclude={"count", "fabric_div", "rf_fabric_div"})
            cfg.update(
                dac=dac,
                tproc_ch=ch,
                fs=fs,
                f_dds=fs / group.interpolation,
                f_fabric=fs / group.fabric_div,
                fdds_div=group.fs_div * group.interpolation,
            )
            gens.append(cfg)
            dacs[dac] = {
                "fs": fs,
                "fs_mult": group.fs_mult,
                "fs_div": group.fs_div,
                "interpolation": group.interpolation,
                "f_fabric": fs / group.rf_fabric_div,
            }
    return gens, dacs


def _readouts(
    profile: _Profile,
    n_gens: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    readouts: list[dict[str, object]] = []
    adcs: dict[str, dict[str, object]] = {}
    for group in profile.readouts:
        fs = group.sample_rate(profile.refclk_freq)
        f_output = fs / (group.decimation * group.samps_per_clk)
        for _ in range(group.count):
            ch = len(readouts)
            adc = f"{ch:02d}"
            cfg = group.model_dump(exclude={"count", "samps_per_clk", "coupling"})
            cfg.update(
                adc=adc,
                tproc_ctrl=n_gens + ch,
                tproc_ch=ch,
                trigger_bit=ch,
                fs=fs,
                f_dds=fs / group.decimation,
                f_output=f_output,
                fdds_div=group.fs_div * group.decimation,
            )
            readouts.append(cfg)
            adcs[adc] = {
                "fs": fs,
                "fs_mult": group.fs_mult,
                "fs_div": group.fs_div,
                "decimation": group.decimation,
                "f_fabric": f_output,
                "coupling": group.coupling,
            }
    return readouts, adcs


def _apply_mux(
    profile: _Profile,
    gens: list[dict[str, object]],
    readouts: list[dict[str, object]],
) -> None:
    gen_groups = [group for group in profile.generators for _ in range(group.count)]
    ro_groups = [group for group in profile.readouts for _ in range(group.count)]
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
        used_gens.add(mux.gen_ch)
        used_readouts.update(mux.ro_chs)
        gen = gens[mux.gen_ch]
        gen_group = gen_groups[mux.gen_ch]
        gen.update(mux.generator.model_dump())
        gen.update(
            f_dds=gen_group.sample_rate(profile.refclk_freq)
            / mux.generator.interpolation,
            fdds_div=gen_group.fs_div * mux.generator.interpolation,
        )
        for port, ch in enumerate(mux.ro_chs):
            ro = readouts[ch]
            ro.pop("tproc_ctrl")
            ro.update(mux.readout.model_dump())
            ro_group = ro_groups[ch]
            fs = ro_group.sample_rate(profile.refclk_freq)
            output_div = ro_group.decimation * (mux.readout.pfb_nch // 2)
            dds_div = (
                output_div if mux.readout.pfb_dds_on_output else ro_group.decimation
            )
            ro.update(
                pfb_port=port,
                trigger_bit=port,
                f_output=fs / output_div,
                f_dds=fs / dds_div,
                fdds_div=ro_group.fs_div * dds_div,
            )


def build_soccfg(profile: Mapping[str, object]) -> QickConfig:
    """Build an isolated compile profile from a YAML-decoded mapping.

    All settings are required; no files, installed-version metadata, hardware or
    environment state are read. See program/README.md for the mapping schema.
    Frequencies are MHz. Groups expand in list order with consecutive DAC/ADC
    names, wave-output ports and ADC trigger bits. Readback inputs use a separate
    channel namespace. RF and logical generator fabric dividers are independent.
    Derived frequency fields must not appear in the input. Invalid, missing or
    redundant fields raise ValueError with their dotted field paths. Neither
    the mapping nor its nested values are mutated or shared with the result.
    """
    settings = _Profile.model_validate(profile)
    gens, dacs = _generators(settings)
    readouts, adcs = _readouts(settings, len(gens))
    _apply_mux(settings, gens, readouts)
    cfg = settings.model_dump(exclude={"generators", "readouts", "rf", "mux"})
    cfg.update(
        gens=gens,
        readouts=readouts,
        rf={**settings.rf.model_dump(), "dacs": dacs, "adcs": adcs},
    )
    return QickConfig(cfg)
