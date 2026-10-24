"""Synthetic mixmux/PFB wiring and a deterministic, non-physical transport."""

import numpy as np
import pytest
from qick import QickConfig
from zcu_tools.cfg_model import MuxReadoutCfg, MuxReadoutTone
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import ModuleCfgFactory, readout


def core_mux(channels=(4, 1), **updates):
    return MuxReadoutCfg(
        gen_ch=0,
        nqz=1,
        mixer_freq=1000.0,
        length=0.537,
        ro_length=0.413,
        pre_delay=0.017,
        post_delay=0.023,
        trig_offset=0.029,
        tones=[
            MuxReadoutTone(freq=1000.0 + 80 * i, gain=0.1, phase=45.0 * i, ro_ch=ch)
            for i, ch in enumerate(channels)
        ],
    ).with_updates(**updates)


def native_mux(channels=(4, 1), **updates):
    try:
        cfg = ModuleCfgFactory.from_core(core_mux(channels, **updates), context=None)
    except ValueError as exc:
        pytest.fail(f"native readout/mux must support from_core and build: {exc}")
    assert isinstance(cfg, readout.MuxReadoutCfg)
    return cfg


def mux_soccfg():
    cfg = make_mock_soccfg(n_gens=2, n_readouts=9).get_cfg()
    cfg["gens"][0].update(
        type="axis_sg_mixmux8_v1",
        has_mixer=True,
        n_tones=8,
        has_gain=True,
        has_phase=True,
        interpolation=8,
        fdds_div=8,
        f_dds=1536.0,
    )
    for i, ro in enumerate(cfg["readouts"][:8]):
        ro.pop("tproc_ctrl")
        ro.update(
            type="axis_pfb_readout_v4",
            ro_type="axis_pfb_readout_v4",
            ro_fullpath="pfb0",
            pfb_port=i,
            pfb_nout=8,
            pfb_nch=64,
            pfb_ch_offset=32,
            pfb_dds_on_output=True,
            has_outsel=False,
            f_output=76.8,
            f_dds=76.8,
            fdds_div=32,
            b_phase=32,
            trigger_bit=i,
        )
    return QickConfig(cfg)


class RecordingProgram(ModularProgramV2):
    """Run the real compiler while recording public scheduling intents."""

    def __init__(self, soccfg, modules, **kwargs):
        self.triggers = []
        self.playbacks = []
        self.pulse_definitions = []
        self.gen_declarations = []
        self.adc_declarations = []
        super().__init__(
            soccfg,
            modules=modules,
            **kwargs,
            reps=2,
        )

    def declare_gen(self, *args, **kwargs):
        self.gen_declarations.append(kwargs.copy())
        return super().declare_gen(*args, **kwargs)

    def declare_readout(self, *args, **kwargs):
        self.adc_declarations.append(kwargs.copy())
        return super().declare_readout(*args, **kwargs)

    def add_pulse(self, *args, **kwargs):
        self.pulse_definitions.append(kwargs.copy())
        return super().add_pulse(*args, **kwargs)

    def trigger(self, ros=None, *args, **kwargs):
        self.triggers.append((list(ros or []), kwargs.copy()))
        return super().trigger(ros, *args, **kwargs)

    def pulse(self, *args, **kwargs):
        self.playbacks.append(kwargs.copy())
        return super().pulse(*args, **kwargs)


class AcquisitionSoc(QickConfig):
    """Only hardware transport plumbing, not mixmux/PFB physics or simulation."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.gen_configs = []
        self.pfb_configs = []

    def start_readout(self, total_shots, counter_addr, ch_list, reads_per_shot):
        self.shots = total_shots
        self.channels = ch_list
        self.reads = reads_per_shot
        self.delivered = False

    def poll_data(self, totaltime=0.1, timeout=None):
        if self.delivered:
            return []
        self.delivered = True
        # Each channel has its own sentinel, multiplied by its rounded window.
        # QICK must normalize this back to [ch+1, -(ch+1)].
        data = []
        for ch, n in zip(self.channels, self.reads, strict=True):
            length = self.us2cycles(0.413, ro_ch=ch)
            data.append(
                np.tile(np.array([ch + 1, -ch - 1]) * length, (self.shots * n, 1))
            )
        return [(self.shots, (data, {}))]

    def config_mux_gen(self, ch, tones):
        self.gen_configs.append((ch, tones))

    def config_mux_readout(self, path, configs, sel):
        self.pfb_configs.append((path, configs, sel))

    def get_tproc_counter(self, addr):
        return 2**31 - 1

    def stop_tproc(self, *args, **kwargs): ...
    def start_tproc(self, *args, **kwargs): ...
    def start_src(self, *args, **kwargs): ...
    def reload_mem(self, *args, **kwargs): ...
    def clear_tproc_counter(self, *args, **kwargs): ...
    def prepare_round(self, *args, **kwargs): ...
    def cleanup_round(self, *args, **kwargs): ...
    def load_bin_program(self, *args, **kwargs): ...
    def load_envelope(self, *args, **kwargs): ...
    def load_weights(self, *args, **kwargs): ...
    def set_nyquist(self, *args, **kwargs): ...
    def set_mixer_freq(self, *args, **kwargs): ...
    def configure_readout(self, *args, **kwargs): ...
    def config_avg(self, *args, **kwargs): ...
    def config_buf(self, *args, **kwargs): ...
    def enable_buf(self, *args, **kwargs): ...
