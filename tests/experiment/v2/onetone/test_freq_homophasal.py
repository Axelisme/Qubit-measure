from __future__ import annotations

import io
import pickle

import numpy as np
import pytest
from zcu_tools.experiment.v2.onetone.freq import (
    FreqCfg,
    FreqExp,
    FreqModuleCfg,
    FreqSweepCfg,
    HomophasalSamplingCfg,
    homophasal_freqs_from_sweep,
)
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, PulseReadoutCfg, SweepCfg
from zcu_tools.program.v2.mocksoc import make_mock_soc, make_mock_soccfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.utils import readout_freq_words


def _theta_from_freq(freqs: np.ndarray, params: HomophasalSamplingCfg) -> np.ndarray:
    return params.theta0 + 2.0 * np.arctan(
        2.0 * params.q_l * (1.0 - freqs / params.r_f)
    )


class _LegacyNumpyUnpickler(pickle.Unpickler):
    """Model a NumPy 1.x server that cannot import NumPy 2's private modules."""

    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            raise ModuleNotFoundError("No module named 'numpy._core'")
        return super().find_class(module, name)


def _homophasal_cfg() -> FreqCfg:
    return FreqCfg(
        reps=1,
        rounds=1,
        modules=FreqModuleCfg(
            readout=PulseReadoutCfg(
                pulse_cfg=PulseCfg(
                    waveform=ConstWaveformCfg(length=1.0),
                    ch=0,
                    nqz=1,
                    freq=6000.0,
                    gain=0.5,
                ),
                ro_cfg=DirectReadoutCfg(
                    ro_ch=0,
                    ro_length=1.0,
                    ro_freq=6000.0,
                    gen_ch=0,
                ),
            )
        ),
        sweep=FreqSweepCfg(
            freq=SweepCfg(start=5960.0, stop=6040.0, expts=9, step=10.0)
        ),
        sampling_mode="homophasal",
        homophasal=HomophasalSamplingCfg(
            r_f=6000.0,
            rf_w=20.0,
            theta0=0.35,
        ),
    )


@pytest.mark.parametrize(
    "sweep",
    [
        SweepCfg(start=5960.0, stop=6040.0, expts=9, step=10.0),
        SweepCfg(start=6040.0, stop=5960.0, expts=9, step=-10.0),
    ],
)
def test_homophasal_freqs_preserve_endpoints_and_equal_theta_spacing(
    sweep: SweepCfg,
) -> None:
    params = HomophasalSamplingCfg(r_f=6000.0, rf_w=20.0, theta0=0.35)

    freqs = homophasal_freqs_from_sweep(sweep, params)

    assert freqs[0] == pytest.approx(sweep.start)
    assert freqs[-1] == pytest.approx(sweep.stop)
    thetas = _theta_from_freq(freqs, params)
    theta_steps = np.diff(thetas)
    np.testing.assert_allclose(theta_steps, np.full_like(theta_steps, theta_steps[0]))


def test_homophasal_sampling_params_require_positive_fit_scale() -> None:
    with pytest.raises(ValueError, match="r_f must be positive"):
        HomophasalSamplingCfg(r_f=0.0, rf_w=20.0, theta0=0.0)

    with pytest.raises(ValueError, match="rf_w must be positive"):
        HomophasalSamplingCfg(r_f=6000.0, rf_w=0.0, theta0=0.0)


def test_readout_freq_words_use_final_generator_and_readout_formats() -> None:
    soccfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    cfg = FreqCfg(
        modules=FreqModuleCfg(
            readout=PulseReadoutCfg(
                pulse_cfg=PulseCfg(
                    waveform=ConstWaveformCfg(length=1.0),
                    ch=1,
                    nqz=1,
                    freq=6000.0,
                    gain=0.5,
                ),
                ro_cfg=DirectReadoutCfg(
                    ro_ch=0,
                    ro_length=1.0,
                    ro_freq=6000.0,
                    gen_ch=1,
                ),
            )
        ),
        sweep=FreqSweepCfg(
            freq=SweepCfg(start=5990.0, stop=6010.0, expts=3, step=10.0)
        ),
    )
    freqs = np.asarray([5990.0, 6000.0, 6010.0], dtype=np.float64)

    gen_words, ro_words = readout_freq_words(
        soccfg,
        freqs,
        gen_ch=1,
        ro_ch=0,
        mixer_freq=None,
        nqz=1,
    )

    assert gen_words == [
        int(soccfg.freq2reg(float(freq), gen_ch=1, ro_ch=0)) for freq in freqs
    ]
    assert ro_words == [
        -int(soccfg.freq2reg_adc(float(freq), ro_ch=0, gen_ch=1)) % 2**32
        for freq in freqs
    ]


def test_homophasal_run_executes_nine_point_mock_soc_path() -> None:
    soc, soccfg = make_mock_soc(n_gens=1, n_readouts=1, sim=None)

    result = FreqExp().run(soc, soccfg, _homophasal_cfg())

    assert result.freqs.shape == (9,)
    assert result.signals.shape == (9,)
    assert np.all(np.diff(result.freqs) > 0.0)
    assert not np.allclose(np.diff(result.freqs), np.diff(result.freqs)[0])


def test_homophasal_run_sends_numpy_version_independent_binprog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soc, soccfg = make_mock_soc(n_gens=1, n_readouts=1, sim=None)
    transmitted_binprogs: list[dict] = []

    def legacy_load_bin_program(binprog: dict, **kwargs) -> None:
        transmitted_binprogs.append(binprog)

    monkeypatch.setattr(soc, "load_bin_program", legacy_load_bin_program)

    FreqExp().run(soc, soccfg, _homophasal_cfg())

    assert len(transmitted_binprogs) == 1
    payload = pickle.dumps(((transmitted_binprogs[0],), {}), protocol=4)
    _LegacyNumpyUnpickler(io.BytesIO(payload)).load()
    assert isinstance(transmitted_binprogs[0]["dmem"], list)
