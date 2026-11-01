# Readout power QND native records

**Readers:** operator, lab-author, framework-dev, agent. Identify the paired-moment power format and the rejected SNR-only format.

The GUI key `twotone/ro_optimize/power` now refers to the paired-readout QND experiment. The native metadata experiment tag changes from `twotone/ge/ro_optimize/gain` to `twotone/ro_optimize/power_qnd`. The new cfg schema starts at `1.0` because this is a different experiment identity.

The old Result stored gain and an SNR score. The new grouped `sums` variable stores float64 data shaped `(Ngain, 8, 3, 3, 3, 3)`. Its NumPy dimension order is Gain, Branch, R1 first moment, R1 second moment, R2 first moment, R2 second moment. Each moment coordinate is `0, 1, 2`, representing `1, I, Q`. The branch coordinate is `prepare_pi*4 + r1_on*2 + middle_pi`. Gain remains in a.u.

The new loader rejects the old native experiment tag before interpreting its variables. No runtime fallback or converter reconstructs paired moments from SNR. The old score does not contain the required information. Keep historical files unchanged and use matching older software to inspect them. The generic container format does not change.

The new cfg drops skew penalty and adds a wait in us. Analysis drops smoothing, gain penalty and automatic optimum selection. It returns gain-wise metrics and Pareto indices. A manually selected optional gain controls writeback without changing the acquired sums.
