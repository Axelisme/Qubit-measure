from qick.asm_v1 import AcquireProgram


def create_waveform(prog: AcquireProgram, name: str, pulse_cfg: dict) -> str:
    ch = pulse_cfg["ch"]
    style = pulse_cfg["style"]

    make_even = False
    if style == "flat_top":
        make_even = True
        pulse_cfg = pulse_cfg["raise_pulse"]

    wav_style = pulse_cfg["style"]
    length = float(pulse_cfg["length"])

    if wav_style == "const":
        if style == "flat_top":
            raise ValueError("Flat top with constant raise style is not supported")
    elif wav_style == "gauss":
        sigma = float(pulse_cfg["sigma"])
        prog.add_gauss(ch, name, sigma, length, even_length=make_even)
    elif wav_style == "cosine":
        prog.add_cosine(ch, name, length, even_length=make_even)
    elif wav_style == "flat_top":
        raise ValueError("Nested flat top pulses are not supported")
    else:
        raise ValueError(f"Unknown waveform style: {wav_style}")


def set_pulse(
    prog: AcquireProgram,
    pulse_cfg: dict,
    ro_ch: int = None,
    waveform: str = None,
):
    ch = pulse_cfg["ch"]
    style = pulse_cfg["style"]
    gain = pulse_cfg["gain"]

    # convert frequency and phase to DAC register values
    freq = prog.freq2reg(pulse_cfg["freq"], gen_ch=ch, ro_ch=ro_ch)
    phase = prog.deg2reg(pulse_cfg["phase"], gen_ch=ch)

    # convert length to cycles
    length = prog.us2cycles(pulse_cfg["length"], gen_ch=ch)

    if style == "gauss" or style == "cosine":
        assert waveform is not None, "Waveform is required for gauss and cosine pulses"
        style = "arb"
        length = None  # already set in waveform
    elif style == "flat_top":
        # the length register for flat_top only contain the flat part
        length = pulse_cfg["length"] - pulse_cfg["raise_pulse"]["length"]
        length = prog.us2cycles(length, gen_ch=ch)
    else:
        raise ValueError(f"Unknown pulse style: {style}")

    prog.set_pulse_registers(
        ch,
        style=style,
        freq=freq,
        phase=phase,
        gain=gain,
        length=length,
        waveform=waveform,
    )
