def create_waveform(prog, ch, pulse_cfg: dict):
    style = pulse_cfg["style"]
    if style == "flat_top":
        raise_cfg = pulse_cfg.setdefault("raise_pulse", {})
        # default raise pulse is 10% of the total length
        raise_cfg.setdefault("length", 0.1 * pulse_cfg["length"])
        # default raise style is cosine
        raise_cfg.setdefault("style", "cosine")
        pulse_cfg = raise_cfg  # use raise pulse for the waveform

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
        sigma = prog.us2cycles(pulse_cfg.get("sigma", pulse_cfg["length"] / 4))
        wavform += f"_S{sigma}"
        prog.add_gauss(ch=ch, name=wavform, sigma=sigma, length=length)
    elif wav_style == "cosine":
        prog.add_cosine(ch=ch, name=wavform, length=length)
    elif wav_style == "flat_top":
        raise ValueError("Nested flat top pulses are not supported")
    else:
        raise ValueError(f"Unknown waveform style: {wav_style}")

    return wavform


def set_pulse(prog, ch, pulse_cfg, waveform=None, for_readout=False, ro=0):
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


def create_pulse(program, ch, pulse_cfg, **kwargs):
    wavform = create_waveform(program, ch, pulse_cfg)

    # set the pulse registers
    set_pulse(program, ch, pulse_cfg, waveform=wavform, **kwargs)
