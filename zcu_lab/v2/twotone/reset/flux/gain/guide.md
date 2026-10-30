**Readers:** operator, agent. Scan a fixed-length flux reset's gain and inspect the remaining excitation.

## Preparation and sequence

Calibrate the qubit pi pulse and averaged-IQ readout at the operating point before this scan.
Each repetition applies optional reset, the init pulse (normally pi), the tested flux reset, then readout.
The tested reset uses `reset/pulse` on `$qubit.wiring.lo_flux_ch`, with frequency 0 MHz and a const waveform.
The gain scan uses a.u.; it does not convert gain to flux.

Open and Reset capture the module sources and fill editable starting values.
Init pulse uses `$qubit.module.x180` when present, otherwise the pi-pulse template; verify the calibration before Run.
Optional reset uses `$qubit.module.reset`; a missing slot leaves it disabled.
Readout uses `$resonator.module.readout`, or the readout template if that slot is absent.
The tested reset starts from the `flux_reset` template with fixed length from `$qubit.ext.flux_reset_length` in us.
Missing required sources remain Invalid. Enter a duration to calibrate gain before a reset length has been saved.

## Scan settings

The gain range starts at a single zero-gain point. Set start, stop and point count for the permitted local-flux range before Run.
Run replaces only the tested pulse's gain; the fixed length and timing remain unchanged.
The tested pulse must retain its const waveform and zero frequency.
Its `pulse_cfg.post_delay` is the time in us to wait after returning from the flux excursion for the flux tail to settle.
0 means no wait. On hardware, measure the flux tail (for example with `fastflux/distortion`) and set this delay long enough for settling.

Reps and rounds start at 1000 and 10.
Relax delay starts at five times the captured T1; missing T1 uses 30 us as a starting time scale, giving 150 us.
Invalid T1 leaves the derived field empty. Review the wait between repetitions rather than treating it as a calibrated reset time.

## Reading the live trace

The live measurement plot shows a real projection of averaged complex IQ versus gain.
Its direction is not calibrated to ground and excited populations, so a low plotted value alone does not establish effective reset.
Retain the raw IQ and compare the response across the scanned range before accepting a calibration.
