# FTF YAML values

**Last updated:** 2026-10-16

`load_ftf(path)` reads UTF-8 YAML once. `parse_ftf(document)` validates an in-memory
document. Both return immutable `FTFSettings`; schema and units are documented
field by field in `schema.py`. Errors identify dotted field paths. Unknown fields,
duplicate bindings, undefined sources, mismatched V/A calibration, invalid source
limits/dimensions and non-PSD receiver noise fail before resource construction.
There is no dependency on `simulate`, `device` or `program`.

The document is `format: zcu.ftf`, `version: 1`. It explicitly supplies physical
model values, native/local/global truncations, channel counts, DC devices/mapping,
pulse calibrations, the common readout source, receiver, two resonators and solver
settings. `zcu_lab/simulation_envs/FTF/default.yaml` is an executable illustrative sample.
It is not a calibrated chip or a convergence guarantee.

Defaults are deliberately limited. Coupler ng, source delay/tau, receiver phase
and offset default to zero. Omitted loss mechanisms are disabled. Omitted resonator
nth uses the common temperature's Bose occupation; explicit zero disables it.
Omitted flux/microwave source lists disable that control family. Omitted seed
means one generated root seed at environment startup. Model energies, couplings,
temperature, truncations, solver tolerances and receiver PSD have no guessed value.

Only generator/readout counts are configurable in the virtual board profile.
Clock, DDS, firmware channel types and memory limits are not YAML knobs. DC and
pulse maps are separate, with F1/C/F2 row order. Microwave coefficient pairs are
real/imaginary GHz per gain multiplying full charge. Resonator couplings also
multiply full charge; no bias-dependent n01 renormalization is performed.

`noise_psd_adc2_ns` is a two-sided white receiver spectral density. The environment
converts it to per-sample covariance by dividing by ADC dt in ns. Resources only
validates its finite symmetric positive-semidefinite 2x2 shape. It does not select
an ADC clock, create a device, evolve a model or decide physical applicability.
