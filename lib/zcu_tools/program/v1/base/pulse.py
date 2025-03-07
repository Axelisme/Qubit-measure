from typing import Any, Dict, Optional


def declare_pulse(prog, pulse: Any, waveform: str, ro_ch: Optional[int] = None):
    prog.declare_gen(pulse["ch"], nqz=pulse["nqz"])
    create_waveform(prog, waveform, pulse)

    times = prog.ch_count[pulse["ch"]]
    assert times > 0, "Something went wrong"
    set_pulse(prog, pulse, ro_ch=ro_ch, waveform=waveform, set_default=(times == 1))


def create_waveform(prog, name: str, pulse_cfg: Dict[str, Any]):
    ch = pulse_cfg["ch"]
    style = pulse_cfg["style"]

    if style == "flat_top":
        pulse_cfg = pulse_cfg["raise_pulse"]

    wav_style = pulse_cfg["style"]
    length = prog.us2cycles(pulse_cfg["length"], gen_ch=ch)

    if style == "flat_top":
        length = int(2 * (length // 2))  # make even

    if wav_style == "const":
        if style == "flat_top":
            raise ValueError("Flat top with constant raise style is not supported")
    elif wav_style == "gauss":
        sigma = prog.us2cycles(pulse_cfg["sigma"], gen_ch=ch)
        prog.add_gauss(ch, name, sigma=sigma, length=length)
    elif wav_style == "drag":
        sigma = prog.us2cycles(pulse_cfg["sigma"], gen_ch=ch)
        delta = pulse_cfg["delta"]
        alpha = pulse_cfg.get("alpha", 0.5)
        prog.add_DRAG(ch, name, sigma=sigma, length=length, delta=delta, alpha=alpha)
    elif wav_style == "cosine":
        prog.add_cosine(ch, name, length=length)
    elif wav_style == "flat_top":
        raise ValueError("Nested flat top pulses are not supported")
    else:
        raise ValueError(f"Unknown waveform style: {wav_style}")


def set_pulse(
    prog,
    pulse_cfg: dict,
    ro_ch: Optional[int] = None,
    waveform: Optional[str] = None,
    set_default=False,
):
    ch = pulse_cfg["ch"]
    style = pulse_cfg["style"]
    gain = pulse_cfg["gain"]

    # convert frequency and phase to DAC register values
    freq = prog.freq2reg(pulse_cfg["freq"], gen_ch=ch, ro_ch=ro_ch)
    phase = prog.deg2reg(pulse_cfg["phase"], gen_ch=ch)

    # convert length to cycles
    length = prog.us2cycles(pulse_cfg["length"], gen_ch=ch)

    kwargs = dict(style=style, freq=freq, phase=phase, gain=gain)

    if style == "const":
        kwargs["length"] = length
    elif style == "flat_top":
        # the length register for flat_top only contain the flat part
        length = pulse_cfg["length"] - pulse_cfg["raise_pulse"]["length"]
        kwargs["length"] = prog.us2cycles(length, gen_ch=ch)
        kwargs["waveform"] = waveform
    elif style in ["gauss", "cosine", "drag", "arb"]:
        assert waveform is not None, f"Waveform is required for {style} pulse"
        kwargs["style"] = "arb"
        kwargs["waveform"] = waveform
    else:
        raise ValueError(f"Unknown pulse style: {style}")

    if set_default:
        prog.default_pulse_registers(ch, **kwargs)
        prog.set_pulse_registers(ch)
    else:
        prog.set_pulse_registers(ch, **kwargs)
