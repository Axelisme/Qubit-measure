**Readers:** operator, agent. Find where resonator drive moves mean readout IQ away from a reference.

# MIST probe-gain trace

Use this scan to locate changes in readout response as you increase the resonator drive. Each shot applies `optional reset → optional qubit preparation → resonator probe → IQ readout → relaxation wait`. The probe carrier and duration stay fixed while its gain changes.

For general operation, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).

## Set the gain window

- Bind the qubit and resonator. **Reset** and **Preparation pulse** start from the bound reset and x180 slots. Missing slots leave those steps off. Select the calibrated pulses for the state you want to prepare.
- Set **Resonator probe** to the intended RF carrier and pulse duration. In **Probe gain sweep** set **Start**, **Stop** and **Expts** (point count). Start in the weak-probe region if you want the default reference to represent weak drive. A descending scan still uses its first points, not the points at the lowest gain.
- Check **Readout** against the resonator calibration. For the Inline fallback, check **Gen ch** (transmit channel), **RO ch** (ADC channel), RF carrier, pulse duration and **RO length** (ADC integration window). Include readout flight time in **Trig offset**, which sets when that window opens. The readout gain stays fixed and this scan does not use the classifier. Leave **Mixer freq** blank only when no digital mixer frequency needs to be set; blank does not turn off an external LO.
- Set **Reps** for shots per gain point in each full sweep and **Rounds** for repeated sweeps averaged in IQ. **Relaxation wait** starts at 5 × captured T1 or 150 µs if T1 is missing. An invalid T1 leaves the wait blank. Set a wait long enough to limit carryover between shots.

## Read the trace

Each point shows `|mean IQ at this gain − reference IQ|` in a.u. Subtract the complex reference before taking the magnitude. This counts displacement in any IQ direction, but it is not a state population and cannot identify a leakage level. Lines connect the measured points; the Analyze figure named `fit` does not contain a model fit.

Live and default Analyze use the complex mean of the first 5% of gain points in scan order, rounded down with a minimum of one point. If that reference is missing, the plot does not substitute a later point. Live reports completed rounds and nominal shots per point. The Analyze summary identifies the source Run and its requested Reps × Rounds when its configuration is available.

The plots have no shot-SEM bars. They show a distance **after** averaging the complex shots, not a mean of per-shot distances or a linear IQ projection. A per-shot standard deviation divided by √N does not give an error bar for this distance. Do not use a small displacement by itself to claim a signal above noise.

Analyze controls use the source Run without taking new data:

- **Ground IQ reference (blank = scan start mean):** supply a reference in a.u. as complex text such as `1+2j`, or leave blank for the scan-start mean.
- **Excited IQ (optional y-limit):** supply a complex IQ in a.u. to set the upper y limit to `1.1 × |Excited IQ − reference IQ|`. This only changes the display; blank uses automatic limits.
- **Photon number / gain squared:** set an independently calibrated AC Stark coefficient in photons per a.u.² to plot `mean photon number = coefficient × gain²` on x. Blank keeps gain on x.

Reanalyzing a loaded mean-IQ trace uses the same reference rule and options; previously saved figures stay as they were.

## Use the response

If only isolated points move, repeat the same window with more shots and check for readout drift. When a change repeats across neighboring gains, narrow the next sweep around those gains. For example, 21 points from 0.12 to 0.14 a.u. give a nominal 0.001 a.u. step. If the change is at an edge, extend that edge within the calibrated range. Use [`singleshot/mist/power`](../../../singleshot/mist/power/guide.md) to check whether populations change there. This scan proposes no writeback.

The `mist_power_trace` recipe starts from a fresh Form even when reusing a tab. Edit **Probe gain sweep** through `overrides`. Omitting an Analyze input keeps its setting, while `null` applies the corresponding blank behavior above. Invalid complex text fails in Analyze after Run; the recipe still attempts Save.
