# Single-fluxonium demo

**Readers:** operator, lab-author. Start an offline single-fluxonium environment and supply a one-point twotone input.

This environment has one fluxonium, one DC source, one microwave line,
one fast-flux line and one readout cavity. It has no coupler, second qubit or mux.
`default.yaml` contains physical settings. `entry-template.yaml` contains working
components, wiring and aliases. Neither supplies a calibrated pulse library.

## Trusted startup

Use this trusted script with the measure GUI's normal launch interface:

```python
connect(launch="new", script="<checkout>/zcu_lab/simulation_envs/single_fluxonium/startup.py")
```

`startup.py` creates or reopens `single-fluxonium-demo/initial`. It installs the
mock SoC and one fixed-unit DC FakeDevice through normal startup facets.
It reads adjacent files, not files relative to the shell cwd. No acquisition,
numeric warmup or real hardware access occurs during startup.

A new entry contains Q1 as `qubit/fluxonium`, R1 as `resonator`, and `flux_bias`
as `device/current_source`. Q1 uses generators 1 and 2. R1 uses generator 0 and
ADC 0. Aliases are `qubit: Q1` and `resonator: $qubit.resonator`.
Flux forms read `$qubit.flux_source` on Open/Reset without a device alias.
The template seeds simulated-truth from
`default.yaml`: Q1's own DC line has `flux_half: 0`, `flux_period: 0.01` and
`flux_int: -0.005` in A. R1 has total linewidth 1.1 MHz and pulse-route flight
time 0.4 us. These values are not measurement results. This environment has
no mux route, global coil or readout amplifier component.
Existing entries keep their edited values and aliases.
Missing files, invalid settings, invalid templates and partial entries raise
instead of opening another environment. Read the startup error before retrying.

The script registers backend cleanup after acquisition drains. GUI registration
transfers each accepted driver to the GUI. Backend cleanup does not close those
drivers. Untransferred drivers remain the script's responsibility. Cleanup still
attempts devices after backend failure and preserves both failures.

## One-point twotone input

Open `twotone/freq`. The entry leaves Q1 frequency and linewidth uncalibrated.
For form initial guesses, set Q1.freq to 580 MHz and Q1.kappa to 5 MHz before
opening the tab, or supply all sweep values explicitly. These are starting
values, not a measured calibration. Supply this complete physical input:

```json
{
  "reps": 100,
  "rounds": 1,
  "relax_delay": 1.0,
  "modules": {
    "reset": null,
    "init_pulse": null,
    "qub_pulse": {
      "type": "pulse", "ch": 1, "nqz": 1,
      "freq": 580.0, "gain": 0.01,
      "waveform": {"style": "const", "length": 0.5}
    },
    "readout": {
      "type": "readout/pulse", "gen_ch": 0, "ro_ch": 0, "nqz": 1,
      "freq": 7000.0, "gain": 0.1,
      "waveform": {"style": "const", "length": 1.0},
      "ro_length": 1.0, "trig_offset": 0.4
    }
  },
  "sweep": {"freq": {"start": 580.0, "stop": 580.0, "expts": 1, "step": 0.0}}
}
```

Module lengths, trigger offset and relax delay use us. Frequencies use MHz.
The 0.4 us trigger offset includes the configured 400 ns receiver delay.
A one-point acquire produces one complex mean IQ value. It does not determine
a transition frequency or support fitting a spectral line. Use a small scan
for those conclusions. Startup assembly and a framework acquire probe do not
replace a separate GUI/MCP acceptance run.

## Readout and replay limits

Only terminal readout is supported. Any earlier pulse on the readout generator
raises ValueError before evolution. The environment does not model its Stark
shift, cavity memory or backaction. Do not treat such a sequence as a completed
simulation. See [field-guide.md](field-guide.md) for basis, bath and line assumptions.

Notebook authors can pass an absolute `private_replay` directory to
`zcu_tools.experiment.single_fluxonium.single_fluxonium`. The resource owner saves
one immutable round reference and detached input content. Call the package's
`replay_round(path)` to recompute that round. These archives contain model truth,
not measured IQ, and belong outside ordinary GUI/RPC metadata.
