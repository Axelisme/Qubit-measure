def create_waveform(program, ch, pulse_cfg):
    style = pulse_cfg["style"]
    length = program.us2cycles(pulse_cfg["length"])

    # add the waveform if needed
    if style == "flat_top":
        wav_style = pulse_cfg["raise_style"]
        wavform = f"{style}_{wav_style}_L{length}"
    else:
        wav_style = style
        wavform = f"{style}_{length}"

    if wav_style == "const":
        pass
    elif wav_style == "gauss":
        sigma = program.us2cycles(pulse_cfg.get("sigma", length / 4))
        wavform += f"_S{sigma}"
        program.add_gauss(ch=ch, name=wavform, sigma=sigma, length=length)
    elif wav_style == "cosine":
        program.add_cosine(ch=ch, name=wavform, length=length)
    else:
        raise ValueError(f"Unknown waveform style: {wav_style}")

    return wavform


def set_pulse(program, ch, pulse_cfg, waveform=None, for_readout=False, ro=0):
    style = pulse_cfg["style"]
    length = program.us2cycles(pulse_cfg["length"])

    # convert frequency and phase to DAC registers
    if for_readout:
        freq_r = program.freq2reg(pulse_cfg["freq"], gen_ch=ch, ro_ch=ro)
    else:
        freq_r = program.freq2reg(pulse_cfg["freq"], gen_ch=ch)
    phase_r = program.deg2reg(pulse_cfg["phase"], gen_ch=ch)

    if style == "const":
        program.set_pulse_registers(
            ch=ch,
            style=style,
            freq=freq_r,
            phase=phase_r,
            gain=pulse_cfg["gain"],
            length=length,
        )
    elif style == "gauss" or style == "cosine" or style == "flat_top":
        program.set_pulse_registers(
            ch=ch,
            style="arb",
            freq=freq_r,
            phase=phase_r,
            gain=pulse_cfg["gain"],
            waveform=waveform,
        )
    else:
        raise ValueError(f"Unknown pulse style: {style}")


def create_pulse(program, ch, pulse_cfg, **kwargs):
    wavform = create_waveform(program, ch, pulse_cfg)

    # set the pulse registers
    set_pulse(program, ch, pulse_cfg, waveform=wavform, **kwargs)
