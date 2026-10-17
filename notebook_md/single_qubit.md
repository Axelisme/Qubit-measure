---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: zcu-tools
    language: python
    name: python3
---

# FTF acquisition with a shared environment

This notebook replaces the former two-level single-qubit simulation. It uses the F1-SQUID-F2 environment, including finite microwave control, a coupler-flux excursion and live bias changes between software rounds. It does not connect to hardware or save into a real result directory.

## Load the environment

Run from the repository root, or supply an absolute environment path. The GUI uses the same Python builder when launched with `--simulate zcu_lab/simulation_envs/FTF/environment.py`. Changing YAML requires restarting the GUI or rebuilding the Notebook environment. A bias change through the existing device controls does not reload YAML.

```python
from pathlib import Path

from zcu_tools.device.fake import FakeDevice
from zcu_tools.experiment.simulation import load_environment
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.utils.device import set_flux
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

environment_path = Path("zcu_lab/simulation_envs/FTF/environment.py").resolve()
env = load_environment(environment_path)
nb_adapter = NotebookAdapter(
    soc=env.soc, soccfg=env.soccfg, device_manager=env.devices,
)
```

`nb_adapter` can wrap caller-selected experiment instances through its normal `run`, `analyze` and persistence APIs. It shares this exact manager; it does not create or reconnect instruments. The acquisition below calls the public program API directly so its waveform and timing are visible without a lab experiment dependency.

## Finite controls and terminal readout

Program lengths and delays are us, frequencies MHz and phases degrees. Program gain is signed normalized DAC amplitude. YAML microwave coefficients multiply the full charge operator in GHz/gain, not a hidden calibrated pi-pulse value. These illustrative pulses are not a gate calibration.

```python
microwave = PulseCfg(
    ch=2, nqz=1, freq=500.0, phase=90.0, gain=-0.1,
    waveform=ConstWaveformCfg(length=0.016),
)
flux = PulseCfg(
    ch=1, nqz=1, freq=0.0, gain=0.2,
    waveform=ConstWaveformCfg(length=0.016), post_delay=0.016,
)
readout = PulseReadoutCfg(
    gen_ch=0, ro_ch=0, nqz=1, freq=6000.0, gain=0.1,
    waveform=ConstWaveformCfg(length=0.08), ro_length=0.1,
    trig_offset=-0.01,
)
program = ModularProgramV2(
    env.soccfg, ProgramV2Cfg(reps=8, rounds=2),
    modules=[microwave.build("microwave"), flux.build("flux"), readout.build("readout")],
)
```

One readout generator and one ADC serve both resonators on the shared reciprocal line. The negative trigger offset includes lookback. The two conditional resonator fields include relaxation and thermal transitions, then the receiver samples one joint initial state and adds I/Q noise for each shot. They are not quantum jump trajectories.

The lowpass tail evolves through the idle time before readout. At the common readout pulse start the model freezes the actual flux, including its residual tail, and uses that Hro, basis, rates and pulls throughout the readout window. It ignores further tail changes in that window. A long tau can therefore change physical accuracy; the model does not claim the tail has vanished or reject it using a settling threshold. Active gate or bias pulses during/after readout are unsupported.

## Change the live DC value between rounds

The initial device value is applied only at startup. The next round reads the current value from the same manager. It does not restore YAML's initial value. The final average mixes the two conditions; this is not automatically a bias sweep.

```python
bias = env.devices.get_device("coupler_bias")
if not isinstance(bias, FakeDevice):
    raise TypeError("This example requires its explicitly built memory source")


def next_bias(round_count, raw, cancel_flag):
    if round_count == 1:
        set_flux(bias, 0.12, unit="V", stop=StopSignal())


averaged_iq = program.acquire(env.soc, progress=False, round_hook=next_bias)
last_round_raw = program.get_raw()
lookback_iq = program.acquire_decimated(env.soc, progress=False)
```

Use I/Q and ordinary analysis to find working points. Normal GUI/MCP metadata exposes only the environment's public capabilities and opaque provenance ID. It does not expose populations, state labels or a calibrated working point. Environment authors may separately call `experiment.ftf.plan_template`, `Engine.evolve` and `populations` to inspect numeric states, outside the operator calibration path.

The YAML example supplies native/local/global dimensions and solver tolerances explicitly. Check larger fixed parent spaces, thermal-state coverage and segment refinement for your sample. This model uses weak dispersive readout, Markov/secular baths and exponential echo-characteristic dephasing. It does not guarantee arbitrary truncations or reproduce a full 1/f Ramsey/echo envelope.

## Close the shared manager

After synchronous work has finished, close the exact returned manager. In a longer notebook, put acquisition in a `try/finally` block so failures also release resources. The GUI owns this cleanup after a successful builder return.

```python
env.devices.close_all_devices()
```
