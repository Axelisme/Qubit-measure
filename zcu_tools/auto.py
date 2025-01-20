from copy import deepcopy
from typing import Union

from .configuration import DefaultCfg
from .tools import deepupdate, numpy2number


def make_cfg(exp_cfg: dict, **kwargs):
    exp_cfg = deepcopy(exp_cfg)
    deepupdate(exp_cfg, kwargs, behavior="force")

    auto_derive(exp_cfg)

    exp_cfg = numpy2number(exp_cfg)

    return exp_cfg


def auto_derive_waveform(pulse_cfg: dict):
    # style and length are required to derive waveform
    if "style" not in pulse_cfg:
        return  # do nothing

    style = pulse_cfg["style"]
    length = pulse_cfg.get("length")
    if style == "flat_top":
        raise_cfg = pulse_cfg.setdefault("raise_pulse", {})
        raise_cfg.setdefault("style", "cosine")
        if length:
            # default raise pulse is 10% of the total length
            # the minimum length is 15 ns
            raise_cfg.setdefault("length", 0.1 * max(length, 0.15))

        # derive raise pulse parameters
        auto_derive_waveform(raise_cfg)
    elif style == "gauss":
        if length:
            # default sigma is 1/4 of the total length
            pulse_cfg.setdefault("sigma", length / 4)


def auto_derive_pulse(name: str, pulse_cfg: Union[str, dict]) -> dict:
    # load pulse configuration if it is a string
    if isinstance(pulse_cfg, str):
        pulse_cfg = deepcopy(DefaultCfg.get_pulse(pulse_cfg))

    ch = None
    nqz = None
    if name == "res_pulse":
        ch = DefaultCfg.get_dac("res_ch")
        nqz = DefaultCfg.get_dac("res_nqz")
    else:
        ch = DefaultCfg.get_dac("qub_ch")
        nqz = DefaultCfg.get_dac("qub_nqz")

    # fill ch if not provided
    if ch is not None:
        pulse_cfg.setdefault("ch", ch)

    # fill nqz if not provided
    if nqz is not None:
        pulse_cfg.setdefault("nqz", nqz)

    # phase
    pulse_cfg.setdefault("phase", 0.0)

    # derive waveform
    auto_derive_waveform(pulse_cfg)

    return pulse_cfg


def auto_derive(exp_cfg):
    dac_cfg = exp_cfg.setdefault("dac", {})
    adc_cfg = exp_cfg.setdefault("adc", {})

    # derive each pulse
    for name, pulse_cfg in dac_cfg.items():
        dac_cfg[name] = auto_derive_pulse(name, pulse_cfg)

    # readout channel
    ro_chs = DefaultCfg.get_adc("ro_chs")
    if ro_chs:
        adc_cfg.setdefault("chs", ro_chs)

    # readout length
    if "res_pulse" in dac_cfg:
        res_pulse = dac_cfg["res_pulse"]
        if "ro_length" in res_pulse:
            adc_cfg.setdefault("ro_length", res_pulse["ro_length"])
        if "length" in res_pulse:
            adc_cfg.setdefault("ro_length", res_pulse["length"])

        if "trig_offset" in res_pulse:
            adc_cfg.setdefault("trig_offset", res_pulse["trig_offset"])

    # trig_offset
    trig_offset = DefaultCfg.get_adc("trig_offset")
    if trig_offset:
        adc_cfg.setdefault("trig_offset", trig_offset)

    # reset
    exp_cfg.setdefault("reset", "none")

    # measure
    exp_cfg.setdefault("readout", "base")

    # flux_dev
    exp_cfg.setdefault("flux_dev", "none")

    # relax delay
    exp_cfg.setdefault("relax_delay", 0.0)
