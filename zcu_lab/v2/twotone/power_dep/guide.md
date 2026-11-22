**Readers:** operator, agent. Find a qubit-probe gain that reveals a transition without obscuring nearby lines.

## Run the scan

At each **Probe gain**, Run sweeps **Probe RF frequency** and measures resonator IQ. A shot applies the optional reset and preparation, plays the qubit probe, reads out the resonator, then waits **Relax delay**. The scan changes the drive setting, not the resonator readout power. Gain in a.u. is a DAC full-scale fraction, not power in dBm.

Bind the qubit and resonator. Open or press **Defaults** to take new starting values; **Refresh** preserves edits. The probe template uses the bound qubit's carrier and generator. The gain sweep starts at 0.001–1.0 a.u. with 101 points; choose a lower maximum if the line already broadens at low gain. Fill or verify the RF endpoints in MHz around a transition measured in `twotone/freq`. Select the calibrated **Readout** from the bound resonator; the template is a starting point when no slot is set. Leave **Reset** blank to omit reset. The default form uses 1000 reps per RF point per round, 10 rounds at each gain and 1 µs of relaxation after each shot.

## Read the map

After **Run** or a successful **Load**, press **Analyze** to redraw the selected run's map and gain spectra. Loading does not open a plot. Analyze shows the source run's **Reps × Rounds** if its settings are available. Older files without those settings still plot, but omit that summary line.

The map gives RF frequency versus probe gain. Color is the magnitude of the mean IQ after subtracting the median complex IQ of the same spectrum. It is in a.u., not a transition probability. Blank cells have no data. During Run, the row counter shows gains with data, including any gain still averaging. The adjacent axes compare the latest gain spectrum with the preceding gain row on that same contrast scale.

Look for a ridge that persists across multiple gain rows. A ridge that shifts or broadens with gain warns against applying the high-gain center or linewidth as a weak-drive calibration. Choose the smallest sampled gain at which the line is reproducible, then use `twotone/freq` to fit a finer scan around the observed center. For example, if the ridge is clear at 0.05 a.u. but has merged with a neighboring line by 0.15 a.u., start the follow-up at 0.05 a.u. rather than 0.15 a.u. If the line reaches the frequency boundary, widen the RF endpoints before comparing widths. This is a two-dimensional spectrum; do not infer shot SEM from the color scale.

Analyze does not fit or propose writeback. Use the map and spectra to choose the gain yourself. This experiment has no dedicated recipe. Save preserves the gain and RF grids with the complex IQ. Loading a saved run does not replace the current form.
