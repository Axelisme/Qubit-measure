**Readers:** operator, agent. Find where resonator drive moves mean readout IQ away from a reference.

# MIST probe-gain trace

Use this scan to locate changes in readout response as you increase the resonator drive. Each shot applies `optional reset → optional qubit preparation → resonator probe → IQ readout → relaxation wait`. The probe carrier and duration stay fixed while its gain changes.

For general operation, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).

## Set up the pulses

Bind the qubit and resonator. **Reset** and **Preparation pulse** start from the bound reset and x180 slots. Missing slots leave those steps off. Select the calibrated pulses for the state you want to prepare.

For inline **Resonator probe** and **Readout** (`Inline: readout/pulse`), set each **Gen ch** to its transmit generator. **Frequency** is the RF target in MHz. Set the probe carrier separately from the readout frequency, which is shared by readout transmission and ADC demodulation. Choose each generator's **NQZ** (Nyquist zone 1 or 2) from the calibration. **Phase** is the transmit pulse phase in degrees, not the Analyze IQ reference. **Mixer freq** is the generator's digital mixer frequency in MHz, not the RF target or the IF. In either section, leave it blank only if the generator needs no mixer frequency setting. Blank does not turn off an external LO.

For each pulse, **Pre-delay** runs from that segment's start to RF playback. **Waveform › Length** sets how long the RF plays, and **Post-delay** starts when playback ends. In **Readout**, **RO ch** selects the ADC and **RO length** sets its integration time. Set **Trig offset** so the ADC window opens when the readout signal reaches it. Measure that offset from the readout segment's start, not from the end of **Pre-delay**. **Post-delay** does not extend the ADC window. All these times are in µs.

Only the probe gain is swept. **Readout › Gain** stays fixed. Use one readout ADC with one integration per shot and no shot mask. The terminal **Classifier**, including its centers, radius and confusion matrix, does not contribute to the mean IQ or set the Ground IQ reference.

## Set the gain window

In **Probe gain sweep**, the visible child fields show how the sweep is set. **Start** and **Stop** mean Range, while **Center** and **Span** specify the midpoint and signed full width. **Expts** is the point count, while **Step** is a signed gain increment. There is no mode dropdown on the sweep row. If a Form printout leaves the parent value blank, read the active child fields instead of using the printed Choices as menu items.

To change which fields you edit, right-click the **Probe gain sweep** heading. Use **Switch to Center / Span** or **Switch to Range** to change coordinates, and **Switch to Step** or **Switch to Expts** to change sampling. Enter valid values in the fields used by the switch first.

With **Start**, **Stop** and **Expts** active, the sweep includes both endpoints. For more than one point the nominal step is `(Stop − Start) / (Expts − 1)` in a.u. A descending sweep has a negative step and still uses its first points for the default reference. Start in the weak-probe region if that reference should represent weak drive.

Set **Reps** for shots per gain point in each full sweep and **Rounds** for repeated sweeps averaged in IQ. **Relaxation wait** starts at 5 × captured T1 or 150 µs if T1 is missing. An invalid T1 leaves the wait blank. Set a wait long enough to limit carryover between shots.

## Read the trace

Each point shows `|mean IQ at this gain − reference IQ|` in a.u. Subtract the complex reference before taking the magnitude. This counts displacement in any IQ direction, but it is not a state population and cannot identify a leakage level. Lines connect the measured points; the Analyze figure named `fit` does not contain a model fit.

Live and default Analyze use the complex mean of the first 5% of gain points in scan order, rounded down with a minimum of one point. If that reference is missing, the plot does not substitute a later point. Live reports completed rounds and nominal shots per point. The Analyze summary identifies the source Run and its requested Reps × Rounds when its configuration is available.

Live updates the error bars after each round. The bars show one shot standard error, estimated by propagating the uncertainty of the mean IQ into the distance. They include projection noise from the retained shots but do not estimate drift between rounds. The scan-start reference uses those same data, so its uncertainty and its correlation with the scan points contribute to the bars. A reference you enter is treated as exact.

The first-order estimate can be too small near zero displacement. At exactly zero, no radial direction is defined, so the bar uses the largest directional standard error of the IQ difference. If the reference is one scan point, that point is compared with itself and its bar is zero. Do not use a small displacement by itself to claim a signal above noise.

A point needs at least two retained shots and their IQ moments for a bar. When the scan-start reference lacks the moments needed to estimate its uncertainty, the bars that depend on it are omitted. A mean-only file still shows the distance, without bars.

Analyze controls use the source Run without taking new data:

- **Ground IQ reference (blank = scan start mean):** supply a reference in a.u. as complex text such as `1+2j`, or leave blank for the scan-start mean.
- **Excited IQ (optional y-limit):** supply a complex IQ in a.u. to set the upper y limit to `1.1 × |Excited IQ − reference IQ|`. This only changes the display; blank uses automatic limits.
- **Photon number / gain squared:** set an independently calibrated AC Stark coefficient in photons per a.u.² to plot `mean photon number = coefficient × gain²` on x. Blank keeps gain on x.

Native Save retains the IQ moments, so reanalyzing the loaded data with the same options reproduces the bars. Previously saved figures stay as they were. Labber export still contains only the mean IQ trace.

## Use the response

If only isolated points move, repeat the same window with more shots and check for readout drift. When a change repeats across neighboring gains, narrow the next sweep around those gains. For example, 21 points from 0.12 to 0.14 a.u. give a nominal 0.001 a.u. step. If the change is at an edge, extend that edge within the calibrated range. Use [`singleshot/mist/power`](../../../singleshot/mist/power/guide.md) to check whether populations change there. This scan proposes no writeback.

The `mist_power_trace` recipe starts from a fresh Form even when reusing a tab. Omit `probe_ref` or `readout_ref` to keep its pulse source. The Form rejects an explicit `null` for either input when selecting the pulse source, before applying `overrides` and before Run. Edit **Probe gain sweep** through `overrides`. Omitting an Analyze input keeps its setting, while `null` applies the corresponding blank behavior above. Invalid complex text fails in Analyze after Run; the recipe still attempts Save.
