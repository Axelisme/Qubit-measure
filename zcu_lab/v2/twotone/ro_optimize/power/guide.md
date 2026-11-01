# Readout gain and QND

**Readers:** operator, agent. Run the paired-readout gain sweep, compare its metrics and choose a gain for writeback.

## Before running

Select the qubit and its resonator. Confirm the qubit pi pulse, readout frequency, readout waveform, integration window and ADC wiring. The optional reset runs before each preparation. The gain sweep overrides the readout gain for both R1 on and R2.

Set `wait` in us. The Form starts it at 0.01 times the captured T1 estimate and starts `relax_delay` at 5 times T1. Missing T1 leaves the corresponding field invalid until you enter a value. These starting values do not establish a suitable photon ring-down time.

`wait` starts after the R1 waveform and its configured post-delay, not after the ADC integration window. Set the readout timing and wait so the two ADC windows do not overlap. Include any extra resonator ring-down time you need in this interval.

## Sequence and stored data

At every gain, the experiment acquires eight branches:

| Branch | Preparation | R1 | Middle pulse |
| --- | --- | --- | --- |
| 0 | I | off | I |
| 1 | I | off | pi |
| 2 | I | on | I |
| 3 | I | on | pi |
| 4 | pi | off | I |
| 5 | pi | off | pi |
| 6 | pi | on | I |
| 7 | pi | on | pi |

Each branch runs reset if enabled, preparation, R1, wait, middle pulse and R2 in that order. I omits the qubit pulse without inserting a pi-length delay. R1 off plays a zero-gain pulse of the same duration and still triggers the ADC. R2 always plays at the swept gain. The paired on/off branches therefore have the same pulse and ADC timing.

The live plot shows the magnitude of the R1 mean-IQ difference between preparation pi and I, using branches 6 and 2. It is a contrast in arbitrary ADC units, not SNR or fidelity.

The Result stores `gains` in a.u. and float64 `sums` shaped `(Ngain, 8, 3, 3, 3, 3)`. The four moment axes are R1, R1, R2, R2, each ordered as `1, I, Q`. The all-zero moment entry counts shots. Reps and completed rounds add to these sums without normalization. Native files store these sums, not raw shots or only the plotted contrast.

## Reading the analysis

Analyze produces a metric grid and an F_sep versus QND scatter plot. The grid shows:

- SNR, Gaussian separation fidelity F_sep and thermal population p from R1-on calibration.
- Retention S, QND Q, upward and downward flip probabilities, and middle-pi miss fraction f, each with a solid R1-on curve and dashed R1-off curve.
- Covariance retention from the R1-on calibration branches as a model check. This curve and the three shared calibration curves have the legend R1 on; they do not have an R1-off estimate.

All metrics are dimensionless. Analysis leaves estimates outside the physical range unclipped. Missing shots or degenerate moments give NaN for the affected metrics. The R1-off curves use the R1-on calibration because a zero-gain R1 has no state separation.

QND includes R1 and the waiting interval, so it also includes ordinary relaxation. Compare on and off before attributing a loss to measurement. A discrepancy between mean-based and covariance retention indicates that the two-state, isotropic-noise model may not describe the data. It does not identify or correct leakage. F_sep is a Gaussian separation estimate, not an assignment fidelity measured from labeled shots.

The scatter colors and labels each point by gain. Its red Pareto frontier retains points for which no other finite point has both at least as large F_sep and QND-on, with one strictly larger. Equal points remain on the frontier. Analysis returns the original gain indices ordered by F_sep. It does not choose an optimum.

## Choosing and saving a gain

Leave `Selected gain (a.u.)` empty to inspect plots without any writeback. Enter a finite gain within `[-1, 1]` and Analyze again to propose that human-selected value. The adapter proposes `$qubit.ext.best_ro_gain`. When the captured best readout frequency and length are also available, it proposes the resonator DPM readout module and its readout slot through the existing writeback path. Missing cfg or calibration coordinates omit the module proposal. Review the proposals before applying them. The chosen value is not snapped to a sweep point.

The GUI experiment key remains `twotone/ro_optimize/power`. Native QND files use tag `twotone/ro_optimize/power_qnd` and cfg schema `1.0`. Load rejects the old SNR-only power tag. Those files lack paired moments and cannot be converted to QND metrics. Keep them for historical analysis with the matching older software.
