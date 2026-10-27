# Six-fluxonium material

**Last updated:** 2026-10-27 (UTC), routing-only residency allowance.

**Readers:** framework-dev, lab-author, agent. Load six circuit materials with local currents and one shared coil.

`parse_six_fluxonium(document)` validates a decoded mapping.
`load_six_fluxonium(path)` reads UTF-8 YAML relative to the caller's cwd.
Both return immutable `SixFluxoniumSettings` and raise `ValueError` with dotted
field diagnostics for invalid material. Neither creates devices or models.

The format is `zcu.six-fluxonium`, strict integer version 1. `qubits.Q1..Q6`
hold chip, microwave, cavity, receiver and numeric material. Each qubit's
`dc_bias` contains its local `source`, `offset`, `period`, `coil_offset` and
`coil_period`. Source settings name `Q1_bias` through `Q6_bias` in circuit order.
Top-level `dc_source` defines `coil_bias`. All seven sources use A, inclusive
limits, startup `initial` current and a positive `ramp_step`.

Both offsets are zero-flux currents in A. Both periods are signed nonzero A per
Phi0. Each circuit's reduced flux is `(I_coil - coil_offset)/coil_period +
(I_local - offset)/period`. Local lines have no crosstalk. Missing fields,
nonfinite values, zero periods, wrong source names and invalid source bounds
fail validation. Negative periods reverse flux direction. The single-fluxonium
schema is unchanged. Old six-system documents have no fallback conversion.
The type contracts and units live in `schema.py`.

Top-level `numerics.routing_cache_bytes` sets a 4 MiB routing-facts allowance
across the environment. Each source receives one sixth, rounded down.
Zero disables routing retention. The value must be a strict nonnegative integer.
The allowance excludes models, complete preparations, scratch and process RSS.
Per-qubit `numerics` continues to hold solver, truncation, seed and pacing values.

Each qubit also accepts optional `drift` with the single-fluxonium schema.
Select `q_cap`, `xqp` or `aphi` with `sigma >= 0` in ln(loss) and
`tau_s > 0` in seconds. Each selected loss must be positive. Omitted parameters
stay constant; omitted/null sections disable drift. The parser rejects invalid
targets before creating resources. Format and version remain unchanged.
The acquisition owner derives independent Q1..Q6 streams from the Q1 root seed
and freezes all losses once per acquisition on the shared simulation clock.

The channel map is fixed. `mux_gen_ch` is 0, `pulse_gen_ch` is 13 and
`pulse_ro_ch` is 6. Q1..Q6 use microwave generators 1..6 and mux ADCs0..5.
There are no fast-flux fields. Unknown sections, missing or
additional qubits, colliding source names and incorrect routes fail validation.
`settings.qubits.ordered()` returns Q1..Q6 independently of YAML mapping order.
