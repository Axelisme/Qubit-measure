# Six-fluxonium material

**Last updated:** 2026-10-23

**Readers:** framework-dev, lab-author. Load six independent circuit materials with one global DC current.

`parse_six_fluxonium(document)` validates a decoded mapping.
`load_six_fluxonium(path)` reads UTF-8 YAML relative to the caller's cwd.
Both return immutable `SixFluxoniumSettings` and raise `ValueError` with dotted
field diagnostics for invalid material. Neither creates devices or models.

The format is `zcu.six-fluxonium`, strict integer version 1. `qubits.Q1..Q6`
hold chip, microwave, cavity, receiver and numeric material. Each qubit's
`dc_bias` contains `offset_phi0` and `flux_per_unit` in Phi0/A.
Top-level `dc_source` defines the one `flux_bias` driver in A, with limits,
initial current and ramp step. The type contracts and units live in `schema.py`.

Top-level `numerics` sets byte allowances for cache residency. Prepare, exact
result and routing totals default to 256, 64 and 4 MiB across the environment.
Each source receives one sixth of each total, rounded down. ADC defaults to
8 MiB per source. Zero disables the corresponding cache. Values must be strict
nonnegative integers. These limits exclude models, scratch and process RSS.
Per-qubit `numerics` continues to hold solver, truncation, seed and pacing values.

The channel map is fixed. `mux_gen_ch` is 0, `pulse_gen_ch` is 13 and
`pulse_ro_ch` is 6. Q1..Q6 use microwave generators 1..6 and mux ADCs0..5.
There are no local DC sources or fast-flux fields. Unknown sections, missing or
additional qubits, colliding source names and incorrect routes fail validation.
`settings.qubits.ordered()` returns Q1..Q6 independently of YAML mapping order.
