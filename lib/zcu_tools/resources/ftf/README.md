# FTF YAML values

**Last updated:** 2026-10-17

`load_ftf(path)` reads UTF-8 YAML once. `parse_ftf(document)` validates an in-memory
document. Both return immutable `FTFSettings`; schema and units are documented
field by field in `schema.py`. Errors identify dotted field paths. Unknown fields,
duplicate bindings, invalid V/A source units, invalid source
limits/dimensions and non-PSD receiver noise fail before resource construction.
There is no dependency on `simulate`, `device` or `program`.

The document is `format: zcu.ftf`, `version: 2`. Version 1 is rejected, without
conversion or compatibility aliases. `chip` supplies physical values, `dc_bias`
combines named device settings and calibration, `drive_lines` binds charge
microwave lines, `readout` groups drive/resonators/ADC, and `numerics` supplies
truncations, solver settings and seed. All loop maps require f1/coupler/f2 keys.
`zcu_lab/simulation_envs/FTF/default.yaml` is an executable illustrative sample,
not a calibrated chip or a convergence guarantee.

Defaults are deliberately limited. Coupler ng, ADC phase and offset default to
zero. Omitted loss mechanisms are disabled. Omitted resonator nth uses the common
temperature's Bose occupation; explicit zero disables it. Omitted or empty
`drive_lines` disables charge control. Omitted seed generates one root seed at
environment startup. Energies, couplings, temperature, truncations, solver
accuracy and ADC PSD have no guessed value.

Generator count is the maximum charge/readout generator index plus one; readout
count is ADC index plus one. Clock, DDS, firmware channel types and memory limits
are fixed. Complex microwave coefficients use real/imag records in GHz per gain
multiplying full charge. Resonator couplings also multiply full charge, without
bias-dependent n01 renormalization. FTF supports static DC bias, charge drive and
readout, not flux pulses. `flux_lines`, including an empty map, is unknown input.

`noise_psd_adc2_ns` is a two-sided white receiver spectral density. The environment
converts it to per-sample covariance by dividing by ADC dt in ns. Resources only
validates its finite symmetric positive-semidefinite 2x2 shape. It does not select
an ADC clock, create a device, evolve a model or decide physical applicability.
