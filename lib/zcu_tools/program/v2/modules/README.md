# program/v2/modules — semantic program modules

**Last updated:** 2026-07-14 — final readout frequency words

High-level cheat-sheet for `program/v2/modules/`. Read before touching this
package. Implementation detail belongs in code and tests; this file records module
roles and design boundaries.

## Purpose

`Module` objects are the semantic building blocks consumed by `ModularProgramV2`.
Each module owns a small piece of program setup in `init()` and time-ordered
runtime emission in `run()`. The module layer should preserve experiment meaning
without leaking hardware register choreography into experiment classes.

## Dmem Tables

- `LoadValue` represents a scalar integer lookup table. It accepts non-negative
  values in the signed-positive int32 range and may bit-pack small values when
  `auto_compress=True`. Use it when later semantic modules interpret the loaded
  register as an ordinary scalar, such as non-uniform delay cycles.
- `LoadWord` represents raw uint32 register words. Its public values use the
  `0..2^32-1` word range; storage maps those words onto the signed int32 dmem
  representation while preserving the hardware bit pattern. It is intentionally
  uncompressed because consumers care about the exact word loaded into a register.

## Readout Runtime Words

- `PulseReadout` accepts optional `gain_val`, `freq_val`, and `ro_freq_val`
  register names when an experiment needs per-point hardware words that cannot be
  represented by a uniform `QickParam` sweep. The module consumes those registers
  explicitly and patches a scratch pulse/readout config before playback; fixed
  readout configs still use the ordinary `PulseReadoutCfg` path.
- Runtime word support is a consumer contract, not a cfg-field contract. The
  persistent readout cfg keeps semantic float/QickParam defaults that are valid as
  templates; raw word ownership remains with the `LoadWord` modules that populate
  the named registers.
- Frequency words supplied to `freq_val` and `ro_freq_val` are the final uint32
  hardware patterns written into wmem. They already include generator mixer
  lowering and the readout downconversion sign; bare `freq2reg` outputs are not
  valid substitutes for this contract.

## Design Boundaries

- Keep scalar values and raw hardware words as distinct module contracts. Expanding
  `LoadValue` to uint32 would make arithmetic/scalar users ambiguous and weaken
  compression assumptions.
- Dmem buffer ownership remains in `ModularProgramV2`; modules append their table
  and keep only the returned offset.
- A module that loads a register does not by itself define how a later pulse,
  readout, or simulator interprets that register. Add explicit consuming module
  contracts for those cases instead of relying on name conventions.
