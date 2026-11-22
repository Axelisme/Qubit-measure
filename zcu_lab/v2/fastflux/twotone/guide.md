**Readers:** operator, agent. Find qubit spectral features versus fast-flux pulse gain and choose the window for a follow-up sweep.

# Fast-flux two-tone spectroscopy

Open `fastflux/twotone`. After optional Reset, a flux pulse and a qubit probe run in parallel. The probe is at the scanned RF frequency. Readout starts after both branches finish. DC flux is not swept. The result locates spectral bands; it does not fit a transition or convert gain to magnetic flux.

## Choose the scan

Bind the intended qubit and resonator before Open. Defaults captures these sources again and discards edits. Reset uses the qubit reset slot or stays disabled. Readout uses the resonator readout slot or the editable template. The Qubit probe uses the qubit microwave drive channel. Check its envelope length and gain. Its carrier comes from Qubit probe frequency, not a fixed module frequency.

Flux pulse starts with the qubit fast-flux channel, a 1 µs rectangular envelope at 0 MHz and Nyquist zone 2. Check channel, timing and safe gain before playing it. The Flux pulse gain scan overrides the module gain. It starts with one point at zero; specify a nontrivial allowed range to map a flux dependence. Gain is the DAC full-scale fraction in a.u.

Qubit probe frequency scans absolute RF in MHz. It starts with 301 points centered on the captured qubit frequency and extends 1.5 linewidths on each side. Enter a wider range if the expected flux shift exceeds that width. Reps defaults to 1000 shots per point per round. Rounds defaults to 10 whole scans. Relax delay starts at 1 µs; set it long enough for repeatable preparation.

## Inspect the resonance map

Run and Analyze show the same gain/frequency heatmap. Color is the real projection of mean complex IQ along the scan's largest variation, in a.u. It is not a state population. The live title counts points with acquired IQ; the round progress shows the accumulation of whole grids. No per-point bars are drawn for a 2D spectrum.

Follow a continuous bright or dark ridge to read the candidate transition frequency at the flux gain of interest. If a feature crosses the high-frequency boundary, increase the frequency stop by at least one observed linewidth. Add enough points to keep the original MHz spacing. If two bands overlap, first narrow the gain range around the crossing and lower the probe gain to check for power-broadened features. Do not assign a transition or change qubit calibration from contrast alone.

Save and Load retain the gain axis, the frequency axis saved in Hz and shown in MHz, complex IQ and the source cfg. Load leaves the current form unchanged. A valid map can still be plotted if historical cfg is missing; the acquisition summary then omits Reps×Rounds. This experiment has no recipe or writeback.
