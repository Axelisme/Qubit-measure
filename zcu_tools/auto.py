from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from .configuration import DefaultCfg
from .tools import deepupdate, numpy2number

NQZ_THRESHOLD = 2000  # MHz


def make_cfg(exp_cfg: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    exp_cfg = deepcopy(exp_cfg)
    deepupdate(exp_cfg, kwargs, behavior="force")

    auto_derive(exp_cfg)

    return numpy2number(exp_cfg)


def auto_derive_waveform(pulse_cfg: Dict[str, Any]):
    # style and length are required to derive waveform
    if "style" not in pulse_cfg:
        return  # do nothing

    style: str = pulse_cfg["style"]
    length: Optional[float] = pulse_cfg.get("length")
    if style == "flat_top":
        raise_cfg: Dict[str, Any] = pulse_cfg.setdefault("raise_pulse", {})
        raise_cfg.setdefault("style", "cosine")
        if length is not None:
            # default raise pulse is 10% of the total length
            # the minimum length is 15 ns
            raise_cfg.setdefault("length", 0.1 * max(length, 0.15))

        # derive raise pulse parameters
        auto_derive_waveform(raise_cfg)
    else:
        if style in ["gauss", "drag"]:
            if length is not None:
                # default sigma is 1/4 of the total length
                pulse_cfg.setdefault("sigma", length / 4)

        if style == "drag":
            pulse_cfg.setdefault("alpha", 0.5)


def auto_derive_pulse(
    name: str, pulse_cfg: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    # load pulse configuration if it is a string
    if isinstance(pulse_cfg, str):
        pulse_cfg = deepcopy(DefaultCfg.get_pulse(pulse_cfg))

    ch = None
    if name == "res_pulse":
        ch = DefaultCfg.get_dac("res_ch")
    else:
        ch = DefaultCfg.get_dac("qub_ch")

    # fill ch if not provided
    if ch is not None:
        pulse_cfg.setdefault("ch", ch)

    # fill nqz if not provided
    if "nqz" not in pulse_cfg:
        if "freq" in pulse_cfg:
            # use NQZ_THRESHOLD as the threshold
            nqz = 2 if pulse_cfg["freq"] > NQZ_THRESHOLD else 1
            pulse_cfg.setdefault("nqz", nqz)

    # phase
    pulse_cfg.setdefault("phase", 0.0)  # deg

    # derive waveform
    auto_derive_waveform(pulse_cfg)

    return pulse_cfg


def auto_derive(exp_cfg: Dict[str, Any]):
    dac_cfg: Dict[str, Any] = exp_cfg.setdefault("dac", {})
    adc_cfg: Dict[str, Any] = exp_cfg.setdefault("adc", {})
    dev_cfg: Dict[str, Any] = exp_cfg.setdefault("dev", {})

    # dac
    ## derive each pulse
    for name, pulse_cfg in dac_cfg.items():
        # check if it is a pulse
        if name.endswith("_pulse"):
            dac_cfg[name] = auto_derive_pulse(name, pulse_cfg)

    ## reset and measure module
    dac_cfg.setdefault("reset", "none")
    dac_cfg.setdefault("readout", "base")

    # adc
    ## readout channel
    ro_chs: Optional[List[int]] = DefaultCfg.get_adc("ro_chs")
    if ro_chs is not None:
        adc_cfg.setdefault("chs", ro_chs)

    ## readout length
    if "res_pulse" in dac_cfg:
        res_pulse: Dict[str, Any] = dac_cfg["res_pulse"]
        if "ro_length" in res_pulse:
            # if pulse cfg has set ro_length, use it
            adc_cfg.setdefault("ro_length", res_pulse["ro_length"])
        if "length" in res_pulse:
            # or, use the length of the readout pulse
            adc_cfg.setdefault("ro_length", res_pulse["length"])

    ## trig_offset
    if "trig_offset" not in adc_cfg:
        # if pulse provide, use it
        res_pulse: Dict[str, Any] = dac_cfg.get("res_pulse", {})
        if "trig_offset" in res_pulse:
            adc_cfg.setdefault("trig_offset", res_pulse["trig_offset"])

        # or, use the timeFly
        timeFly: Optional[float] = DefaultCfg.get_adc("timeFly")
        if timeFly is not None:
            adc_cfg.setdefault("trig_offset", timeFly)

    ## relax delay
    adc_cfg.setdefault("relax_delay", 0.0)  # us

    # dev
    ## flux dev
    flux_dev: Optional[str] = DefaultCfg.get_dev("flux_dev")
    if flux_dev is not None:
        # if default flux_dev is provided, use it
        dev_cfg.setdefault("flux_dev", flux_dev)
    # use none if not provided
    dev_cfg.setdefault("flux_dev", "none")

    # other
    flux_value: Optional[float] = DefaultCfg.get_dev("flux")
    if flux_value is not None:
        dev_cfg.setdefault("flux", flux_value)

    # round to soft_avgs
    if "rounds" in exp_cfg:
        exp_cfg["soft_avgs"] = exp_cfg["rounds"]
