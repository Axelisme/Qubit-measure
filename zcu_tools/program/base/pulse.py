from typing import Union

from qick.asm_v1 import AcquireProgram


def create_waveform(prog: AcquireProgram, ch: int, pulse_cfg: dict) -> str:
    style = pulse_cfg["style"]
    if style == "flat_top":
        # use raise pulse for the waveform
        pulse_cfg = pulse_cfg["raise_pulse"]

    wav_style = pulse_cfg["style"]
    length = prog.us2cycles(pulse_cfg["length"])

    if style == "flat_top":
        length = 2 * (length // 2)  # make length even
        wavform = f"flatTop_{wav_style}_L{length}"
    else:
        wavform = f"{wav_style}_L{length}"

    if wav_style == "const":
        if style == "flat_top":
            raise ValueError("Flat top with constant raise style is not supported")
    elif wav_style == "gauss":
        # default sigma is quarter of the length
        sigma = prog.us2cycles(pulse_cfg["sigma"])
        wavform += f"_S{sigma}"
        prog.add_gauss(ch=ch, name=wavform, sigma=sigma, length=length)
    elif wav_style == "cosine":
        prog.add_cosine(ch=ch, name=wavform, length=length)
    elif wav_style == "flat_top":
        raise ValueError("Nested flat top pulses are not supported")
    else:
        raise ValueError(f"Unknown waveform style: {wav_style}")

    return wavform


def set_pulse(
    prog: AcquireProgram,
    ch: int,
    pulse_cfg: dict,
    waveform: str = None,
    for_readout=False,
    ro=0,
):
    style = pulse_cfg["style"]
    length = prog.us2cycles(pulse_cfg["length"])

    # convert frequency and phase to DAC registers
    if for_readout:
        freq_r = prog.freq2reg(pulse_cfg["freq"], gen_ch=ch, ro_ch=ro)
    else:
        freq_r = prog.freq2reg(pulse_cfg["freq"], gen_ch=ch)
    phase_r = prog.deg2reg(pulse_cfg["phase"], gen_ch=ch)

    if style == "const":
        prog.set_pulse_registers(
            ch=ch,
            style=style,
            freq=freq_r,
            phase=phase_r,
            gain=pulse_cfg["gain"],
            length=length,
        )
    elif style == "gauss" or style == "cosine":
        prog.set_pulse_registers(
            ch=ch,
            style="arb",
            freq=freq_r,
            phase=phase_r,
            gain=pulse_cfg["gain"],
            waveform=waveform,
        )
    elif style == "flat_top":
        raise_length = prog.us2cycles(pulse_cfg["raise_pulse"]["length"])
        raise_length = 2 * (raise_length // 2)  # make length even
        flat_length = length - raise_length
        prog.set_pulse_registers(
            ch=ch,
            style="flat_top",
            freq=freq_r,
            phase=phase_r,
            gain=pulse_cfg["gain"],
            length=flat_length,
            waveform=waveform,
            stdysel="zero",
        )
    else:
        raise ValueError(f"Unknown pulse style: {style}")


def is_single_pulse(pulse_cfg: dict):
    # use gain key to determine if the pulse is single pulse or nested pulse
    if "gain" in pulse_cfg:
        # gain should be a number
        assert not isinstance(pulse_cfg["gain"], dict), "Invalid pulse configuration"
        return True
    # only one level of nesting is supported
    assert all(["gain" in v for v in pulse_cfg.values()]), "Invalid pulse configuration"
    return False


def create_pulse(
    prog: AcquireProgram, ch: int, pulse_cfg: dict, **kwargs
) -> Union[str, dict]:
    if is_single_pulse(pulse_cfg):
        # single pulse
        wavform = create_waveform(prog, ch, pulse_cfg)
        set_pulse(prog, ch, pulse_cfg, wavform, **kwargs)
    else:
        # nested pulse
        wavform = {k: create_waveform(prog, ch, v) for k, v in pulse_cfg.items()}
        # don't need to set pulse for nested pulse
    return wavform
