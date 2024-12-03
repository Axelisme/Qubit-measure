from qick.asm_v1 import AcquireProgram


def is_single_pulse(pulse_cfg: dict):
    # use style key to determine if the pulse is single pulse or nested pulse
    if "style" in pulse_cfg:
        # style should be a string
        assert not isinstance(pulse_cfg["style"], dict), "Invalid pulse configuration"
        return True
    # only one level of nesting is supported
    assert all(
        ["style" in v for v in pulse_cfg.values()]
    ), "Invalid pulse configuration"
    return False


def create_waveform(prog: AcquireProgram, ch: int, pulse_cfg: dict) -> str:
    def create_one(prog: AcquireProgram, ch: int, pulse_cfg: dict):
        style = pulse_cfg["style"]
        if style == "flat_top":
            # use raise pulse for the waveform
            pulse_cfg = pulse_cfg["raise_pulse"]

        wav_style = pulse_cfg["style"]
        length = prog.us2cycles(pulse_cfg["length"], gen_ch=ch)

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
            sigma = prog.us2cycles(pulse_cfg["sigma"], gen_ch=ch)
            wavform += f"_S{sigma}"
            prog.add_gauss(ch=ch, name=wavform, sigma=sigma, length=length)
        elif wav_style == "cosine":
            prog.add_cosine(ch=ch, name=wavform, length=length)
        elif wav_style == "flat_top":
            raise ValueError("Nested flat top pulses are not supported")
        else:
            raise ValueError(f"Unknown waveform style: {wav_style}")

        return wavform

    if is_single_pulse(pulse_cfg):  # single pulse
        return create_one(prog, ch, pulse_cfg)
    # nested pulse
    return {k: create_one(prog, ch, v) for k, v in pulse_cfg.items()}


def set_pulse(
    prog: AcquireProgram,
    pulse_cfg: dict,
    gen_ch: int,
    ro_ch: int = None,
    waveform: str = None,
):
    style = pulse_cfg["style"]
    length = prog.us2cycles(pulse_cfg["length"], gen_ch=gen_ch)

    # convert frequency and phase to DAC registers
    freq_r = prog.freq2reg(pulse_cfg["freq"], gen_ch=gen_ch, ro_ch=ro_ch)
    phase_r = prog.deg2reg(pulse_cfg["phase"], gen_ch=gen_ch, ro_ch=ro_ch)

    if style == "const":
        prog.set_pulse_registers(
            ch=gen_ch,
            style=style,
            freq=freq_r,
            phase=phase_r,
            gain=pulse_cfg["gain"],
            length=length,
        )
    elif style == "gauss" or style == "cosine":
        prog.set_pulse_registers(
            ch=gen_ch,
            style="arb",
            freq=freq_r,
            phase=phase_r,
            gain=pulse_cfg["gain"],
            waveform=waveform,
        )
    elif style == "flat_top":
        raise_length = pulse_cfg["raise_pulse"]["length"]
        raise_length = prog.us2cycles(raise_length, gen_ch=gen_ch)
        raise_length = 2 * (raise_length // 2)  # make length even
        flat_length = length - raise_length
        assert flat_length >= 0, "Raise pulse length is longer than the total length"
        prog.set_pulse_registers(
            ch=gen_ch,
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
