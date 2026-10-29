# Integer-fluxonium environment

**Readers:** operator, lab-author, agent. Start the offline zero-flux environment and measure transition frequencies before preparing higher states.

This environment uses the Mencia 2024 Device D circuit inputs
EJ/EC/EL = 6.78/1.47/0.22 GHz, zero flux and a 20 mK qubit bath.
They are synthetic inputs, not laboratory calibration.
The model has one fluxonium, two charge-drive lines and one terminal cavity
readout. It has no fast-flux route, coupler, mux or real hardware connection.

## Trusted startup and working copy

From the checkout root, create your local settings file:

```bash
cp zcu_lab/simulation_envs/integer_fluxonium/default.yaml zcu_lab/simulation_envs/integer_fluxonium/integer_fluxonium.yaml
```

Edit `integer_fluxonium.yaml`, then start a new measure GUI with this trusted
script through the normal launch interface:

```python
connect(launch="new", script="<checkout>/zcu_lab/simulation_envs/integer_fluxonium/startup.py")
```

Startup reads the working copy beside the script, independent of the shell cwd.
A missing working copy raises `FileNotFoundError` with its absolute path and a
copy command before creating an entry. Startup never falls back to `default.yaml`.
Restart the GUI to load settings changes. Existing working copies do not update
when the tracked template changes.

Startup creates or reopens `integer-fluxonium-demo/initial` under the selected
project's `result/` and `Database/` roots.
A new entry receives Q1, R1, a `flux_bias` current-source component and aliases
`qubit: Q1`, `resonator: $qubit.resonator`.
The schema-version-3 template leaves Q1.freq null and supplies no pulse library.
Existing entries retain edited parameters, library contents and aliases.
Partial entries, invalid templates and invalid settings raise errors rather
than switching environments. Older component documents need the explicit
[upgrade command](../../../docs/parameter-upgrades.md).

The script installs a simulated SoC and one fixed-unit DC FakeDevice using the
same lifecycle as single-fluxonium. It performs no acquisition or numeric warmup.
After operations drain, the close callback closes the backend and releases
untransferred devices. Drivers accepted by the GUI belong to the GUI and close
there. Backend cleanup failure does not skip device cleanup.

Use a separate project root for this environment. Do not mix synthetic results
with hardware results or reuse another environment's entry as calibration.
Neither GUI nor MCP should read simulator truth, internal populations or replay
archives. Use the acquired IQ and visible spectra as measurement evidence.

## Channels and units

| Route | Generator | Meaning |
| --- | ---: | --- |
| Readout | 0 | Cavity drive, ADC 0 |
| Pump | 1 | Full charge operator, 1 GHz/gain |
| Stokes | 2 | Full charge operator, 1 GHz/gain |

Q1.wiring.ch supplies the first drive channel. Enter the second channel explicitly
in experiments that need it. The shared qubit wiring schema is unchanged.
Generator 2 is not a fast-flux channel. Pulse routing rejects a generator absent
from the working settings' `drive_lines` before evolution. Add a named charge
line to those settings and restart before using a different channel.

Physical YAML uses GHz and ns. Experiment Forms use MHz and us.
Thus a 400 ns readout is 0.4 us. The readout reference settings are 7460 MHz,
gain 0.1, a constant 0.4 us waveform, 0.4 us acquisition and zero propagation
delay. Enter these settings explicitly or save a reviewed readout module in
R1.module.readout. The generic readout Form template has its own duration and
trigger margin; it does not load this environment's physical YAML.
Cavity coupling 0.10 GHz is distinct from charge-drive calibration 1 GHz/gain.

## Transition spectroscopy

1. Open `twotone/freq` from New Tab. Q1.freq is null, so Frequency Start/Stop
   remain blank and Invalid. Enter an operator-chosen search window and review
   the probe, readout, averaging and timing settings. No transition frequency
   comes from the simulator model.
2. Keep Init Pulse at None. Find 0–1 from ground, refine the line and record its
   measured frequency. Search for 0–3 from ground in a separate scan.
   Keep each transition's calibration separate.
3. Calibrate π03 using the existing Rabi amplitude or length experiment with
   the measured 0–3 carrier. Store the reviewed pulse in the library, or retain
   its frequency, gain, waveform and channel for inline entry.
4. In `twotone/freq`, change Init Pulse from None to that library pulse or Inline.
   Enter the calibrated π03 values. Keep its carrier fixed and sweep Probe Pulse
   to find 3–2 after preparation. Run plays reset, π03, swept probe, then readout.
   Init Pulse None preserves the original ground-state sequence.
5. Use measured 0–3 and 3–2 carriers for Pump and Stokes in subsequent experiments.
   Do not Apply those fits to Q1.freq as though they were the 0–1 transition.
   For a Raman center, use spectroscopy to estimate the 0–2 splitting rather
   than reading the model spectrum.

The [spectroscopy guide](../../v2/twotone/freq/guide.md) explains fitting and
writeback. After editing source parameters or library slots, Reset recaptures
them and may replace unsaved Form inputs. Refresh preserves authored inputs.
A visible Form is not proof of a valid cfg or a calibrated preparation.

## Numerical scope

The default basis is oscillator cutoff 80 with 20 retained DC eigenstates.
Raman's full-drive comparison required this increase from 15 levels.
The new default passed the five independent-seed reference readout checks
and bounded STIRAP acquisition, endpoint and sampled-peak regressions.
The historical seven reference-only basis checks used 80/15 against a finite
100/25 reference. They are not a new whole-grid validation at 80/20.
Separate STIRAP characterization records acquired preparation-label transfer,
amplitude response and basis comparisons at fixed seed and controls.
The [field guide](field-guide.md) records those values and thresholds.
It keeps the acquired-IQ results separate from lab-author-only environment
oracle comparisons of basis size, full drive and finite-time level-3 samples.
Operators, GUI and MCP use acquired IQ, never those oracle populations.

The STIRAP evidence applies only to its recorded working points, controls,
seeds, times and numerical budgets. It does not establish convergence over
an entire scan grid, at stronger drives or for Raman, and it does not measure
experimental gate fidelity. The field guide separately records Raman's
12-point acquired-IQ and full-drive numerical limits, plus single-point
80/15 and 80/20 acquisition wall times. Existing working YAML files do not
inherit a changed template; review their basis and restart after editing.
The fixed circuit's synthetic f03 is about 7.24 GHz rather than the literature's
6.86 GHz. Do not retune the circuit to match that number, and do not copy it
into a Form or entry as calibration.
