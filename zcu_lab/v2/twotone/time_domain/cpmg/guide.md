## Measure CPMG coherence

**Readers:** operator, agent. CPMG measures coherence decay for several pi-pulse counts. Each sequence applies pi/2, a train of n refocusing pi pulses, a final pi/2 projection and readout. Calibrate pi, pi/2 and readout before running. The sequence corrects the end intervals for finite pulse widths. Intervals too short to fit the pulses cause Run to fail; increase the minimum time.

**Last updated:** 2026-10-21

## Configure the scan

Pi pulse counts accepts Log or List input. Log uses a maximum count and requested point count. It starts at 1 and retains the maximum count. It rounds geometrically spaced values to integers and removes duplicates. A maximum greater than 1 requires at least two requested points. A maximum of 1 accepts one point and returns `[1]`. Use List for any other single count. The resulting list may have fewer points than requested. List accepts a nonempty JSON array of positive integers, such as `[1, 2, 4, 8]`. Zero, negative counts and fractional counts are invalid.

Minimum time and Maximum time define one shared total-time window in us for every pulse count. Time points sets the samples per row. The initial values are 1 us, 100 us and 101 points. Increase the minimum time when a large pulse count leaves too little spacing between pulses. The same point count applies to every pulse count.

Detune ratio is the number of fringe periods per time index. The default is 0.1. Use Fit fringes for this phase-ramped acquisition. Set Detune ratio to 0 and disable Fit fringes for a decay-only acquisition. Early-stop SNR optionally ends averaging of each row when its signal is resolved.

## Module sources and waiting

The form uses the captured `$qubit.module.x180`, `$qubit.module.x90` and `$qubit.module.readout` slots. Missing slots use pi_pulse, pi2_pulse and readout templates. Reset uses `$qubit.module.reset` and stays disabled when absent. Missing calibration in a template remains Invalid until supplied.

Relax delay starts at 5 times the captured `$qubit.t1`. A missing or null T1 uses 30 us, giving a 150 us wait. An invalid stored T1 leaves the derived input empty. Open and Reset capture these sources. After alias or calibration edits, use Reset to refresh the form.

## Run and analyze

Run displays normalized IQ versus pulse count and time. Save the raw data, then Analyze to fit a decay for each count. The fit figure compares accepted T_CPMG values with pulse count on log axes. Optional Ramsey T2 and T1 references add horizontal T2R and 2*T1 lines in us; the fit uses the measured decay independently of these references.

Incomplete rows or rejected fits have null T2 values in the summary. Analysis fails when every row fails its fit checks. Extend the time window if the decay is unresolved. Review each accepted T_CPMG value; calibration stays unchanged.

The cpmg recipe follows the same Run, save and Analyze steps. Use max_n with points for Log input, or times for an explicit list. min_length_us, max_length_us and length_points set the shared time window. Reusing a tab resets it before applying overrides.
